import openai
import pybullet as pb
import numpy as np
import optas
import sys
from time import sleep,perf_counter
import threading
import multiprocessing as mp
from scipy.spatial.transform import Rotation as Rot

import re


class VisualBox:
    def __init__(
        self,
        base_position,
        half_extents,
        rgba_color=[0, 1, 0, 1.0],
        base_orientation=[0, 0, 0, 1],
    ):
        visid = pb.createVisualShape(
            pb.GEOM_BOX, rgbaColor=rgba_color, halfExtents=half_extents
        )
        self._id = pb.createMultiBody(
            baseMass=0.0,
            basePosition=base_position,
            baseOrientation=base_orientation,
            baseVisualShapeIndex=visid,
        )

    def reset(self, base_position, base_orientation=[0, 0, 0, 1]):
        pb.resetBasePositionAndOrientation(
            self._id,
            base_position,
            base_orientation,
        )

class Controller:

    def __init__(self, dt):
        self.response_gain = 0.4
        ee_link = 'storz_tilt_endoscope_link_cm_optical'
        T = 2
        robot = optas.RobotModel(urdf_filename='resources/lbr_with_tilt_endoscope.urdf', time_derivs=[0, 1])
        name = robot.get_name()
        robot.add_base_frame('pybullet_world', [1, 0, 0])
        builder = optas.OptimizationBuilder(T, robots=robot)
        qc = builder.add_parameter('qc', robot.ndof)
        box_pos = builder.add_parameter('box_pos', 3)
        insertion_depth = builder.add_parameter('depth', 1)
        axis_align = builder.add_parameter('axis_align', 3)
        # response = builder.add_parameter('response', 3)
        eff_goal = box_pos + insertion_depth*axis_align
        q0 = builder.get_model_state(name, 0)
        dq = builder.get_model_state(name, 0, time_deriv=1)
        qF = builder.get_model_state(name, 1)
        builder.add_equality_constraint('initial', q0, qc)
        builder.add_equality_constraint('dynamics', q0 + dt*dq, qF)
        builder.enforce_model_limits(name)

        pc = robot.get_global_link_position(ee_link, qc)

        TfF = robot.get_global_link_transform(ee_link, qF)
        pF = TfF[:3, 3]
        zF = TfF[:3, 2]

        # Minimize distance from end-effector position to trocar axis
        alpha = optas.dot(pF - box_pos, axis_align)
        dist_to_axis_align_sq = optas.sumsqr(box_pos + alpha*axis_align - pF)
        builder.add_cost_term('dist_to_axis_align', 1e3*dist_to_axis_align_sq)

        # Minimize distance between end-effector position and goal
        dist_to_goal_sq = optas.sumsqr(eff_goal - pF)
        task_weight = 1.  # TODO: consider dyanmic weight (i.e. weight becomes stronger when dist to axis align is smaller)
        alpha = optas.dot(pc - box_pos, axis_align)
        dist_to_axis_align_sq = optas.sumsqr(box_pos + alpha*axis_align - pc)
        task_weight = optas.exp(-dist_to_axis_align_sq**2) # TODO: does this need to be squared?
        builder.add_cost_term('dist_to_goal', 50.*task_weight*dist_to_goal_sq)

        # Align end-effector with trocar axis
        builder.add_cost_term('eff_axis_align', 50.*optas.sumsqr(zF - axis_align))

        # Minimize joint velocity
        W = 0.1*optas.diag([7, 6, 5, 4, 3, 2, 1])
        builder.add_cost_term('min_dq', dq.T@W@dq)

        # Reduce interaction forces
        delta = 0.5
        pR = pc #+ delta*response
        builder.add_cost_term('reduce_interaction', 100*optas.sumsqr(pF - pR))

        self.solver = optas.ScipyMinimizeSolver(builder.build()).setup('SLSQP')
        self.name = name
        self.solution = None

        c = optas.sumsqr(pF - eff_goal)
        self._finish_criteria = optas.Function('finish', [qF, box_pos, axis_align], [c])

    def finish_criteria(self, q, bp, z):
        tol = 0.005**2
        f = self._finish_criteria(q, bp, z)
        return f < tol

    def __call__(self, qc, bp, z, d):

        # resp = self.response_gain*f
        # print(resp)
        if self.solution is not None:
            self.solver.reset_initial_seed(self.solution)
        else:
            self.solver.reset_initial_seed({f'{self.name}/q': optas.horzcat(qc, qc)})
        self.solver.reset_parameters({'qc': qc, 'box_pos': bp, 'axis_align': z, 'depth':d})
        t0 = perf_counter()
        self.solution = self.solver.solve()
        t1 = perf_counter()
        dur = t1 - t0
        self.dur = dur
        # print(f"-------\nSolver duration (s, hz): {dur}, {1/dur}")
        return self.solution[f'{self.name}/q'][:, -1].toarray().flatten()

    def get_solver_duration(self):
        return self.dur


class Robot:
    def __init__(self,time_step):
        self.time_step = time_step
        self.id = pb.loadURDF(
            'resources/lbr_with_tilt_endoscope.urdf',
            basePosition=[1, 0, 0],
        )
        self._dq = np.zeros(7)
        self._robot = optas.RobotModel(urdf_filename='resources/lbr_with_tilt_endoscope.urdf', time_derivs=[0, 1])
        self._robot.add_base_frame('pybullet_world', [1, 0, 0])
        # self._J = self._robot.get_global_geometric_jacobian_function('lbr_link_ee')
        self._J = self._robot.get_global_geometric_jacobian_function('storz_tilt_endoscope_link_cm_optical')
        self._p = self._robot.get_global_link_position_function('storz_tilt_endoscope_link_cm_optical')
        self._Tf = self._robot.get_global_link_transform_function('storz_tilt_endoscope_link_cm_optical')

    def Tf(self):
        return self._Tf(self.q()).toarray()

    def p(self):
        return self._p(self.q()).toarray().flatten()

    def J(self, q):
        return self._J(q).toarray()

    @property
    def num_joints(self):
        return pb.getNumJoints(self.id)

    @property
    def joint_indices(self):
        return list(range(self.num_joints))

    @property
    def joint_info(self):
        joint_info = []
        for joint_index in self.joint_indices:
            joint_info.append(pb.getJointInfo(self.id, joint_index))
        return joint_info

    @property
    def joint_types(self):
        return [info[2] for info in self.joint_info]

    @property
    def revolute_joint_indices(self):
        return [
            joint_index
            for joint_index, joint_type in zip(self.joint_indices, self.joint_types)
            if joint_type == pb.JOINT_REVOLUTE
        ]

    @property
    def ndof(self):
        return len(self.revolute_joint_indices)

    def update(self):
        self._dq = self.dq()

    def q(self):
        return np.array([state[0] for state in pb.getJointStates(self.id, self.revolute_joint_indices)])

    def dq(self):
        return np.array([state[1] for state in pb.getJointStates(self.id, self.revolute_joint_indices)])

    def ddq(self):
        return (self.dq() - self._dq)/self.time_step

    def tau(self):
        return np.array([state[3] for state in pb.getJointStates(self.id, self.revolute_joint_indices)])

    def tau_ext(self):
        tau = self.tau()
        q = self.q().tolist()
        dq = self.dq().tolist()
        ddq = self.ddq().tolist()
        tau_ext = np.array(pb.calculateInverseDynamics(self.id, q, dq, ddq)) - tau
        # lim = 1.1*np.array([-7.898,  3.47 , -1.781, -5.007,  1.453,  1.878,  2.562])
        lim = 2*np.ones(7)

        for i in range(7):
            if -lim[i] <= tau_ext[i] < lim[i]:
                tau_ext[i] = 0.

        return tau_ext

    def f_ext(self):
        J = self.J(self.q())
        tau_ext = self.tau_ext()
        J_inv = np.linalg.pinv(J, rcond=0.05)
        f_ext = J_inv.T @ tau_ext
        f_ext *= np.array([0.01, 0.01, 0.01, 1, 1, 1])

        return f_ext

    def reset(self, q, deg=False):
        for joint_index, joint_position in zip(self.revolute_joint_indices, q):
            pb.resetJointState(self.id, joint_index, joint_position)

    def cmd(self, q):
        pb.setJointMotorControlArray(
            self.id,
            self.revolute_joint_indices,
            pb.POSITION_CONTROL,
            q,
        )

    def cmd_vel(self, dq):
        pb.setJointMotorControlArray(
            self.id,
            self.revolute_joint_indices,
            pb.VELOCITY_CONTROL,
            dq,
        )


class HRI:
    def __init__(self):
        # self.thread_rob = threading.Thread(target=self.robot_loop)

        # self.thread_hri = threading.Thread(target=self.hri_loop)
        self.manager = mp.Manager()
        self.shared_dict = self.manager.dict()

        self.depth = 0.3
    
    @ staticmethod
    def get_box_pose(box, noise=0.005):
        p, r = pb.getBasePositionAndOrientation(box._id)
        noisep = np.random.uniform(-noise, noise, size=(len(p),))
        noiser = np.random.uniform(-noise, noise, size=(len(r),))
        return np.array(p) + noisep, np.array(r) + noiser
    
    def run(self):

        input_thread = threading.Thread(target=self.hri_loop)
        input_thread.daemon = True  
        input_thread.start()

        self.robot_loop()
        

    def robot_loop(self):
        #robotic simulation setup
        if 'gui' in sys.argv:
            connect_args = [pb.GUI]
        else:
            connect_args = [pb.DIRECT]

        pb.connect(
            *connect_args
        )

        pb.resetSimulation()
        gravz = -9.81
        pb.setGravity(0, 0, gravz)

        sampling_freq = 240
        time_step = 1./float(sampling_freq)
        pb.setTimeStep(time_step)

        box_base_position = np.array([0.35, -0.2, 0.2])

        pb.resetDebugVisualizerCamera(
            cameraDistance=0.2,
            cameraYaw=-40,
            cameraPitch=30,
            cameraTargetPosition=box_base_position,
        )
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.setPhysicsEngineParameter(sparseSdfVoxelSize=0.25)
        
        robot = Robot(time_step)
        q0 = np.deg2rad([-40, -45, 0, 90, 0, -45, 0])
        robot.reset(q0)
        robot.cmd(q0)
        
        

        init_pos = np.array([0.2,0.2,0.3])
        controller = Controller(time_step)


        # environment setup
        box_half_extents = [0.01, 0.01, 0.01]
        constdist = optas.np.array([0.0, 0.0, -0.04])
        box_id1 = VisualBox(
            base_position=init_pos + constdist,
            half_extents=box_half_extents,
        )

        box_id2 = VisualBox(
            base_position=np.array([-0.3,-0.3,0.3]) + constdist,
            half_extents=box_half_extents,
        )

        t = 0.
        rest_wait_time = 5.  # secs
        time_factor_for_rest = 0.0001

        # wait for everything is ready
        while pb.isConnected():
            pb.stepSimulation()
            sleep(time_step*time_factor_for_rest)
            t += time_step
            robot.update()
            if t > rest_wait_time:
                break
        # self.isLoopRun = True
        print("!!!System has been setup")
        # initial task
        
        depth = self.depth
        box_id = box_id1

        time_factor = 0.000001
        q = q0

        # loop for robotic simulation
        # while(pb.isConnected() and self.isLoopRun):
        print("Run into robot_loop")
        while True:
            box_base_position, box_base_orientation = HRI.get_box_pose(box_id)
            R_box = Rot.from_quat(box_base_orientation).as_matrix()

            goal_z = -R_box[:3, 1]

            q = controller(q, box_base_position, goal_z, self.depth)
            robot.cmd(q)

            pb.stepSimulation()
            sleep(time_step*time_factor)
            robot.update()

        pb.disconnect()

    def hri_loop(self):
        # Provide your OpenAI API key
        openai.api_key = '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'

        pattern = r"@{(-?\d+\.\d+)}"
        print("Here is chatgpt Thread")
        pre_command = "assume you are a robotic controller, when I tell you move closer to my target. \
        You directly give me a sentence like \"0.5\". No explains, only you can reply is a number from -1.0 to 1.0.\
        Further means number close to 1.0 or -1.0. Closer means number close to 1.0\
        if I say \"send it\", you need to output the number in such format: @{0.3}"
        conversation_history = ""
        conversation_history += f"User: {pre_command}\nAssistant: "

        response=chat_with_gpt(conversation_history)
        print("ChatGPT: " + response)
        # Example usage
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break
            # chat_history.append(user_input)
            conversation_history += f"User: {user_input}\nAssistant: "
            response = chat_with_gpt(conversation_history)
            print("ChatGPT: " + response)

            match = re.search(pattern, response)
            if match:
                self.depth = float(match.group(1))
                print("Robot: Command has been sent")





            
            






def chat_with_gpt(prompt):
    messages = [
        {"role": "system", "content" : prompt}
        ]
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    if 'choices' in response and len(response.choices)>0:
        return response.choices[0].message['content']
    else:
        return None
    
def main():

    # setup pybullet
    

    instance = HRI()
    instance.run()



    # # Provide your OpenAI API key
    # openai.api_key = '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'

    # # chat_with_gpt("Target Position is")

    # # Example usage
    # while True:
    #     user_input = input("User: ")
    #     if user_input.lower() == 'exit':
    #         break
    #     response = chat_with_gpt(user_input)
    #     print("ChatGPT: " + response)


if __name__ == "__main__":
    sys.exit(main())