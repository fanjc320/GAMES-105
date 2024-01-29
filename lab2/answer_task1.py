import numpy as np
import copy
import math
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
# ------------- lab1里的代码 -------------#
def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1,3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order)+ ''.join(rot_order)

            elif 'Frame Time:' in line:
                break
        
    joint_parents = [-1]+ [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets

def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#

'''
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
'''

class BVHMotion():
    def __init__(self, bvh_file_name = None) -> None:
        
        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []
        
        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        #! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None # (N,M,4)的ndarray, 用四元数表示的局部旋转
        
        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass
    
    #------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)
        
        motion_data = load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros((motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros((motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                continue   
            elif self.joint_channel[i] == 3:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:, cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]
            self.joint_rotation[:, i, :] = R.from_euler('XYZ', rotation,degrees=True).as_quat()
            cur_channel += self.joint_channel[i]
        
        return

    def batch_forward_kinematics(self, joint_position = None, joint_rotation = None): # 对25个关节一起操作
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation
        
        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向
        
        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:,pi,:]) 
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])  # 子关节的位移 等于父关节位置+朝向 * 关节位置!!!!!!
            joint_orientation[:, i, :] = (parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation
    
    
    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name) for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name) for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:,idx,:]
        self.joint_rotation = self.joint_rotation[:,idx,:]
        pass
    
    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)
    
    @property
    def motion_length(self):
        return self.joint_position.shape[0]
    
    
    def sub_sequence(self, start, end):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end,:,:]
        res.joint_rotation = res.joint_rotation[start:end,:,:]
        return res
    
    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate((self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate((self.joint_rotation, other.joint_rotation), axis=0)
        pass
    
    #--------------------- 你的任务 -------------------- #

    # @staticmethod
    # def decompose_rotation_with_yaxis(rotation): # 原本根节点的旋转quartanion
    #     '''
    #     输入: rotation 形状为(4,)的ndarray, 四元数旋转
    #     输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
    #     '''
    #     euler = R.from_quat(rotation).as_euler('XYZ', degrees=True)
    #     print("decompose_rotation_with_yaxis rotation euler:", euler)
    #     Ry = np.zeros_like(rotation)
    #     Rxz = np.zeros_like(rotation)
    #     # TODO: 你的代码
    #     rotation_matrix = R.from_quat(rotation).as_matrix()
    #     R1 = rotation_matrix[:, 1] # ???? 旋转后的y轴在固定坐标系三个坐标轴xyz上的投影分量，或者说新的单位y轴在固定空间的坐标；y为上，这里是拿到"上"的朝向。
    #     y_axis = np.array([0., 1., 0.]) # 老空间，即固定坐标系的 y轴
    #     rot_axis = np.cross(R1, y_axis) # 叉乘得到转轴，注意是表示新y轴向量到老y轴的旋转，而非相反,
    #     print("decompose_rotation_with_yaxis rotation_matrix:", rotation_matrix)
    #     print("decompose_rotation_with_yaxis R1:", R1)
    #     print("decompose_rotation_with_yaxis rot_axis:", rot_axis) # rot_axis: [-0.14020839  0.          0.10673311]
    #     theta = np.arccos(np.dot(R1, y_axis) / np.linalg.norm(R1)) # 新老y轴的夹角
    #     if theta == 0.:   #?????
    #         return [1., 0., 0., 0.], rotation
    #     # 长度等于旋转角 https://zhuanlan.zhihu.com/p/93563218
    #     R_prime = R.from_rotvec(theta * rot_axis / np.linalg.norm(rot_axis)) #从新老y轴的夹角和转轴得到旋转向量，代表从老y轴(及固定坐标的y轴)到新y轴(rotation的y轴)的旋转
    #     Ry = (R_prime * R.from_quat(rotation)).as_quat()#????
    #     print("decompose_rotation_with_yaxis R_prime euler:", R_prime.as_euler('XYZ', degrees=True))
    #     print("decompose_rotation_with_yaxis Ry euler:", R.from_quat(Ry).as_euler('XYZ', degrees=True))
    #     Rxz = (R.from_quat(Ry).inv() * R.from_quat(rotation)).as_quat() #????
    #     print("decompose_rotation_with_yaxis Rxz euler:", R.from_quat(Rxz).as_euler('XYZ', degrees=True))
    #     return Ry, Rxz

    @staticmethod
    # def decompose_rotation_with_yaxis(self, rotation):
    def decompose_rotation_with_yaxis(rotation):
        '''
        输入: rotation 形状为(4,)的ndarray, 四元数旋转
        输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
        '''
        # TODO: 你的代码
        # 将四元数旋转分解为绕y轴的旋转，和转轴在xz平面的旋转，先得到Ry，再逆运算得到Rxz
        Ry = R.from_quat(rotation).as_euler("XYZ", degrees=True)
        print("decompose_rotation_with_yaxis Ry euler:", Ry) # decompose_rotation_with_yaxis Ry euler: [45. 45. 45.]
        Ry = R.from_euler("XYZ", [0, Ry[1], 0], degrees=True)
        print("decompose_rotation_with_yaxis Ry euler:", Ry.as_euler("XYZ", degrees=True)) # decompose_rotation_with_yaxis Ry euler: [ 0. 45.  0.]

        Rxz = Ry.inv() * R.from_quat(rotation)
        # print("decompose_rotation_with_yaxis Rxz euler:", R.from_quat(Rxz).as_euler('XYZ'))
        print("decompose_rotation_with_yaxis Rxz euler:", Rxz.as_euler("XYZ", degrees=True)) # decompose_rotation_with_yaxis Rxz euler: [30.3611934   8.42105812 75.3611934 ]
        return Ry, Rxz

    # part 1
    def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz):
        '''
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            你需要完成并使用decompose_rotation_with_yaxis
            输入的target_facing_direction_xz的norm不一定是1
        '''
        
        res = self.raw_copy() # 拷贝一份，不要修改原始数据
        
        # 比如说，你可以这样调整第frame_num帧的根节点平移
        offset = target_translation_xz - res.joint_position[frame_num, 0, [0,2]]
        res.joint_position[:, 0, [0,2]] += offset
        # TODO: 你的代码
        # 把第frame_num帧的根节点的face转到target
        Ry, _ = self.decompose_rotation_with_yaxis(res.joint_rotation[frame_num, 0, :])# 第frame_mum帧，根节点的旋转即朝向;Ry是木偶人y轴的旋转,可通过矩阵取z拿到朝向!!
        print("translation_and_rotation Ry euler:", Ry.as_euler('XYZ', degrees=True)) # Ry是rotation # translation_and_rotation Ry euler: [ 0.       -8.135959  0.      ]
        # print("translation_and_rotation Ry euler:", R.from_quat(Ry).as_euler('XYZ', degrees=True)) # Ry是quartanion
        # translation_and_rotation Ry euler: [0.         0.71361138 0.]
        rot_target = np.array([target_facing_direction_xz[0], 0, target_facing_direction_xz[1]]) ### 旋转的方向中y轴旋转，在xz面转向，方向向量只有x,z
        # source是Ry的z轴，target是目标的z轴，旋转轴是y轴!!!!!!!!!!!!
        # rot_source = R.from_quat(Ry).as_matrix()[:, 2] # 旋转矩阵可看作新空间向量(1,1,1)，在老空间三个坐标轴的投影，现在只取z的投影,因为把z轴看做前方!!!!!
        rot_source = Ry.as_matrix()[:, 2] ### 只去z轴的朝向，因为z代表"前方", 应该是木偶人根节点y轴在世界坐标下的z轴的投影,即木偶人的朝向,Ry是木偶人根节点y轴朝向.
        Ry_matrix = Ry.as_matrix()
        print("translation_and_rotation Ry_matrix:", Ry_matrix)
        rot_target = rot_target / np.linalg.norm(rot_target)
        rot_source = rot_source / np.linalg.norm(rot_source)
        rot_axis = np.cross(rot_source, rot_target) # 旋转轴通过叉乘得到, 用来判断y旋转的正负
        # Ry_matrix = R.from_quat(Ry).as_matrix()
        # print("translation_and_rotation Ry_matrix:", Ry_matrix)
        print("translation_and_rotation rot_source:", rot_source)
        print("translation_and_rotation rot_target:", rot_target)
        print("translation_and_rotation rot_axis:", rot_axis)

        # decompose_rotation_with_yaxis rotation euler: [-1.425206 -8.135959 -1.141766]
        # decompose_rotation_with_yaxis rotation_matrix: [[ 0.98973848  0.0197257  -0.14152255]
        #  [-0.01640085  0.9995623   0.02462164]
        #  [ 0.14194628 -0.02204789  0.98962879]]
        # decompose_rotation_with_yaxis R1: [ 0.0197257   0.9995623  -0.02204789]
        # decompose_rotation_with_yaxis rot_axis: [ 0.02204789 -0.          0.0197257 ]
        # decompose_rotation_with_yaxis R_prime euler: [1.26335354 0.01246198 1.13027269]
        # decompose_rotation_with_yaxis Ry euler: [-1.01777750e-13 -8.15001747e+00 -5.08888749e-14]
        # decompose_rotation_with_yaxis Rxz euler: [-1.41085846  0.011571   -0.93974146]
        # translation_and_rotation Ry_matrix: [[ 9.89900278e-01  1.03735476e-15 -1.41765439e-01]
        #  [-8.61258471e-16  1.00000000e+00  1.30352479e-15]
        #  [ 1.41765439e-01 -1.16826287e-15  9.89900278e-01]]
        # translation_and_rotation rot_source: [-1.41765439e-01  1.30352479e-15  9.89900278e-01]
        # translation_and_rotation rot_target: [0.70710678 0.         0.70710678]
        # translation_and_rotation rot_axis: [ 9.21731218e-16  8.00208502e-01 -9.21731218e-16]
        #

        # 虽然理论上rot_axis就是y轴，但是float的精度问题，必须重新设置为y轴，不然人物会飞
        if rot_axis[1] > 0.:   # 判断y旋转的正负, 是向左走，还是向右走
            rot_axis = np.array([0., 1., 0.])
        else:
            rot_axis = np.array([0., -1., 0.])
        theta = np.arccos(np.dot(rot_source, rot_target)) #
        delta_rotation = R.from_rotvec(theta * rot_axis) # 旋转向量，方向为y

        # 修改orientation
        res.joint_rotation[:, 0, :] = np.apply_along_axis(lambda q: (delta_rotation * R.from_quat(q)).as_quat(), axis=1,
                                                          arr=res.joint_rotation[:, 0, :]) # 对所有关节修改朝向

        look_apply_along_axis = np.apply_along_axis(lambda q: q, axis=1,
                                                          arr=res.joint_rotation[:, 0, :])

        print("translation_and_rotation res.joint_rotation[:, 0, :].shape:", res.joint_rotation[:, 0, :].shape, " res.joint_rotation[:, 0, :]:", res.joint_rotation[:, 0, :])
        print("translation_and_rotation look_apply_along_axis.shape:", look_apply_along_axis.shape, " look_apply_along_axis:", look_apply_along_axis)
        look_apply_along_axis_0 = np.apply_along_axis(lambda q: q, axis=0,
                                                    arr=res.joint_rotation[:, 0, :])
        print("translation_and_rotation look_apply_along_axis_0.shape:", look_apply_along_axis_0.shape,
              " look_apply_along_axis_0:", look_apply_along_axis_0)
        # 修改position
        offset_center = res.joint_position[frame_num, 0, [0, 2]] # 修改根节点的x,z的位置，让其"移动"
        res.joint_position[:, 0, [0, 2]] -= offset_center
        res.joint_position[:, 0, :] = np.apply_along_axis(delta_rotation.apply, axis=1, arr=res.joint_position[:, 0, :])
        res.joint_position[:, 0, [0, 2]] += offset_center

        return res

def get_interpolate_pose(rot1, rot2, pos1, pos2, w):
    # 这个slerp的参数times到底是什么意思
    ret_rot = np.empty_like(rot1)
    for i in range(len(rot1)):
        slerp = Slerp([0, 1], R.from_quat([rot1[i], rot2[i]])) #?????
        ret_rot[i] = slerp([w]).as_quat()

    # 使用欧拉角直接线性插值，会产生鬼畜
    # interpolate_euler = (1 - w) * R.from_quat(rot1).as_euler('XYZ') + w * R.from_quat(rot2).as_euler('XYZ')
    # ret_rot = R.from_euler('XYZ', interpolate_euler).as_quat()

    ret_pos = (1 - w) * pos1 + w * pos2
    return ret_rot, ret_pos
# part2
def blend_two_motions(bvh_motion1, bvh_motion2, alpha):
    '''
    blend两个bvh动作
    假设两个动作的帧数分别为n1, n2
    alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作应该有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    '''
    
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros((len(alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((len(alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[...,3] = 1.0

    # TODO: 你的代码
    for i in range(len(alpha)):
        cur_time = i / (len(alpha) - 1.)

        frame_num1 = len(bvh_motion1.joint_rotation)
        time1 = cur_time * (bvh_motion1.motion_length - 1)
        index1 = math.floor(time1)
        alpha1 = time1 - index1
        rotation1, position1 = get_interpolate_pose(\
            bvh_motion1.joint_rotation[index1, ...], \
            bvh_motion1.joint_rotation[(index1 + 1) % frame_num1, ...], \
            bvh_motion1.joint_position[index1, ...], \
            bvh_motion1.joint_position[(index1 + 1) % frame_num1, ...], \
            alpha1)

        frame_num2 = len(bvh_motion2.joint_rotation)
        time2 = cur_time * (bvh_motion2.motion_length - 1)
        index2 = math.floor(time2)
        alpha2 = time2 - index2
        rotation2, position2 = get_interpolate_pose(\
            bvh_motion2.joint_rotation[index2, ...], \
            bvh_motion2.joint_rotation[(index2 + 1) % frame_num2, ...], \
            bvh_motion2.joint_position[index2, ...], \
            bvh_motion2.joint_position[(index2 + 1) % frame_num2, ...], \
            alpha2)

        res.joint_rotation[i, ...], res.joint_position[i, ...] = get_interpolate_pose(\
            rotation1, rotation2, position1, position2, alpha[i])

    return res

# part3
def build_loop_motion(bvh_motion):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res = bvh_motion.raw_copy()
    
    from smooth_utils import build_loop_motion
    return build_loop_motion(res)

def nearest_frame(motion, target_pose):
    def pose_distance(pose1, pose2):
        total_dis = 0.
        for i in range(1, pose1.shape[0]):
            total_dis += np.linalg.norm(pose1[i] - pose2[i])
        return total_dis

    min_dis = float("inf")
    ret = -1
    for i in range(motion.motion_length):
        dis = pose_distance(motion.joint_rotation[i], target_pose)
        print(dis)
        if dis < min_dis:
            ret = i
            min_dis = dis
    return ret
# part4
# def concatenate_two_motions(bvh_motion1, bvh_motion2, mix_frame1, mix_time):
#     '''
#     将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
#     混合开始时间是第一个动作的第mix_frame1帧
#     虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
#     Tips:
#         你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
#     '''
#     res = bvh_motion1.raw_copy()
#
#     # TODO: 你的代码
#     # 下面这种直接拼肯定是不行的(
#     # res.joint_position = np.concatenate([res.joint_position[:mix_frame1], bvh_motion2.joint_position], axis=0)
#     # res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1], bvh_motion2.joint_rotation], axis=0)
#     # 将bvh2设置为循环动作
#     bvh_motion2 = build_loop_motion(bvh_motion2)
#     motion = bvh_motion2
#     pos = motion.joint_position[-1, 0, [0, 2]]
#     rot = motion.joint_rotation[-1, 0]
#     facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
#     new_motion = motion.translation_and_rotation(0, pos, facing_axis)
#     bvh_motion2.append(new_motion)
#
#     start_frame2 = nearest_frame(bvh_motion2, bvh_motion1.joint_rotation[mix_frame1])
#     translation_xz = bvh_motion1.joint_position[mix_frame1, 0, [0, 2]]
#     Ry, _ = bvh_motion1.decompose_rotation_with_yaxis(bvh_motion1.joint_rotation[mix_frame1, 0, :])
#     facing_direction_xz = R.from_quat(Ry).as_matrix()[2, [0, 2]]
#     facing_direction_xz = [0, 1.0]
#     bvh_motion2 = bvh_motion2.translation_and_rotation(start_frame2, translation_xz, facing_direction_xz)
#
#     cur_mix_frame1 = mix_frame1
#     cur_mix_frame2 = start_frame2
#     for i in range(mix_time):
#         cur_mix_frame1 += 1
#         cur_mix_frame2 += 1
#         res.joint_rotation[cur_mix_frame1], res.joint_position[cur_mix_frame1] = get_interpolate_pose( \
#             res.joint_rotation[cur_mix_frame1],
#             bvh_motion2.joint_rotation[cur_mix_frame2],
#             res.joint_position[cur_mix_frame1],
#             bvh_motion2.joint_position[cur_mix_frame2],
#             (i + 1.) / mix_time)
#
#     res.joint_position = np.concatenate(
#         [res.joint_position[:cur_mix_frame1], bvh_motion2.joint_position[cur_mix_frame2:]], axis=0)
#     res.joint_rotation = np.concatenate(
#         [res.joint_rotation[:cur_mix_frame1], bvh_motion2.joint_rotation[cur_mix_frame2:]], axis=0)
#
#     return res

# https://github.com/Cltsu/GAMES105/blob/main/lab2/answer_task1.py
# Decomposing a matrix(用于分解变换矩阵至旋转、平移，缩放分量) https://blog.csdn.net/GISsirclyx/article/details/4730543

# part4
def concatenate_two_motions(bvh_motion1, bvh_motion2, mix_frame1, mix_time):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧,
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()

    # TODO: 你的代码
    # 下面这种直接拼肯定是不行的(
    # 从mix_frame1截断， 先播放frame1，再播放frame2，这种肯定是不对的，最起码对动作进行一个转换，对吧

    # 从mix_frame开始的动作到新动作的第一帧对齐
    rot = bvh_motion1.joint_rotation[mix_frame1, 0] # 第mix_frame1(60)帧的根节点旋转
    facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]] # z轴被旋转后，在固定空间的投影，取x,z ????

    new_bvh_motion2 = bvh_motion2.translation_and_rotation(0, bvh_motion1.joint_position[mix_frame1, 0, [0, 2]],
                                                           facing_axis)

    # 进行动画blending,分别使用惯性混合和线性插值方法
    blending_joint_position = np.zeros(
        (mix_time, new_bvh_motion2.joint_position.shape[1], new_bvh_motion2.joint_position.shape[2]))
    blending_joint_rotation = np.zeros(
        (mix_time, new_bvh_motion2.joint_rotation.shape[1], new_bvh_motion2.joint_rotation.shape[2]))
    blending_joint_rotation[..., 3] = 1.0 # 统一归一化为单位quartanion

    # 惯性方法： inertialize
    half_time = 0.3
    dt = 1 / 60
    y = 4.0 * 0.69314 / (half_time + 1e-5) # ln2 = 0.693147

    from smooth_utils import quat_to_avel
    src_avel = quat_to_avel(bvh_motion1.joint_rotation[mix_frame1 - 15:mix_frame1], dt)
    dst_avel = quat_to_avel(new_bvh_motion2.joint_rotation[0:15], dt)
    off_avel = src_avel[-1] - dst_avel[0]
    off_rot = (R.from_quat(bvh_motion1.joint_rotation[mix_frame1]) * R.from_quat(
        new_bvh_motion2.joint_rotation[0].copy()).inv()).as_rotvec()

    src_vel = bvh_motion1.joint_position[mix_frame1] - bvh_motion1.joint_position[mix_frame1 - 1]
    dst_vel = new_bvh_motion2.joint_position[1] - new_bvh_motion2.joint_position[0]
    off_vel = (src_vel - dst_vel) / 60
    off_pos = bvh_motion1.joint_position[mix_frame1] - new_bvh_motion2.joint_position[0]

    for i in range(len(new_bvh_motion2.joint_position)): # (100,25,3)
        tmp_ydt = y * i * dt
        eydt = np.exp(-tmp_ydt)
        # eydt = 1.0 / (1.0 + tmp_ydt + 0.48 * tmp_ydt * tmp_ydt + 0.235 * tmp_ydt * tmp_ydt * tmp_ydt)
        j1 = off_vel + off_pos * y
        j2 = off_avel + off_rot * y
        off_pos_i = eydt * (off_pos + j1 * i * dt)
        off_vel_i = eydt * (off_vel - j1 * y * i * dt)
        off_rot_i = R.from_rotvec(eydt * (off_rot + j2 * i * dt)).as_rotvec()
        off_avel_i = eydt * (off_avel - j2 * y * i * dt)

        new_bvh_motion2.joint_position[i] = new_bvh_motion2.joint_position[i] + off_pos_i
        new_bvh_motion2.joint_rotation[i] = (
                    R.from_rotvec(off_rot_i) * R.from_quat(new_bvh_motion2.joint_rotation[i])).as_quat()

    # # 线性blending，动画增加30帧
    # for i in range(mix_time):
    #     t = i / mix_time
    #     blending_joint_position[i] = (1-t) * res.joint_position[mix_frame1] + t * new_bvh_motion2.joint_position[0]
    #     for j in range(len(res.joint_rotation[mix_frame1])):
    #         blending_joint_rotation[i, j] = slerp(res.joint_rotation[mix_frame1,j], new_bvh_motion2.joint_rotation[0,j], t)
    # new_bvh_motion2.joint_position = np.concatenate([blending_joint_position,  new_bvh_motion2.joint_position], axis=0)
    # new_bvh_motion2.joint_rotation = np.concatenate([blending_joint_rotation,  new_bvh_motion2.joint_rotation], axis=0)

    res.joint_position = np.concatenate([res.joint_position[:mix_frame1], new_bvh_motion2.joint_position], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1], new_bvh_motion2.joint_rotation], axis=0)

    return res

def test_decompose_rotation_with_yaxis():
    # rotation = R.from_euler('XYZ', [0,0,45], degrees=True)
    # decompose_rotation_with_yaxis rotation euler: [ 0.  0. 45.]
    # decompose_rotation_with_yaxis rotation_matrix: [[ 0.70710678 -0.70710678  0.        ]
    #  [ 0.70710678  0.70710678  0.        ]
    #  [ 0.          0.          1.        ]]
    # decompose_rotation_with_yaxis R1: [-0.70710678  0.70710678  0.        ]
    # decompose_rotation_with_yaxis rot_axis: [ 0.          0.         -0.70710678]
    # decompose_rotation_with_yaxis R_prime euler: [  0.   0. -45.]
    # decompose_rotation_with_yaxis Ry euler: [0. 0. 0.]
    # decompose_rotation_with_yaxis Rxz euler: [ 0.  0. 45.]
    # -------------------------------------------
    rotation = R.from_euler('XYZ', [45,0,0], degrees=True)
    # decompose_rotation_with_yaxis rotation euler: [45.  0.  0.]
    # decompose_rotation_with_yaxis rotation_matrix: [[ 1.          0.          0.        ]
    #  [ 0.          0.70710678 -0.70710678]
    #  [ 0.          0.70710678  0.70710678]]
    # decompose_rotation_with_yaxis R1: [0.         0.70710678 0.70710678]
    # decompose_rotation_with_yaxis rot_axis: [-0.70710678  0.          0.        ]
    # decompose_rotation_with_yaxis R_prime euler: [-45.   0.   0.]
    # decompose_rotation_with_yaxis Ry euler: [0. 0. 0.]
    # decompose_rotation_with_yaxis Rxz euler: [45.  0.  0.]
    # --------------------------------------------
    # rotation = R.from_euler('XYZ', [0,45,0], degrees=True)
    # decompose_rotation_with_yaxis rotation euler: [ 0. 45.  0.]
    # decompose_rotation_with_yaxis rotation_matrix: [[ 0.70710678  0.          0.70710678]
    #  [ 0.          1.          0.        ]
    #  [-0.70710678  0.          0.70710678]]
    # decompose_rotation_with_yaxis R1: [0. 1. 0.]
    # decompose_rotation_with_yaxis rot_axis: [0. 0. 0.]
    # --------------------------------------------
    rotation = R.from_euler('XYZ', [45, 45, 45], degrees=True)
    # decompose_rotation_with_yaxis rotation euler: [45. 45. 45.]
    # decompose_rotation_with_yaxis rotation_matrix: [[ 0.5        -0.5         0.70710678]
    #  [ 0.85355339  0.14644661 -0.5       ]
    #  [ 0.14644661  0.85355339  0.5       ]]
    # decompose_rotation_with_yaxis R1: [-0.5         0.14644661  0.85355339]
    # decompose_rotation_with_yaxis rot_axis: [-0.85355339  0.         -0.5       ]
    # decompose_rotation_with_yaxis R_prime euler: [-66.87499616  21.85509089 -32.59645168]
    # decompose_rotation_with_yaxis Ry euler: [ 0.         29.27761319  0.        ]
    # decompose_rotation_with_yaxis Rxz euler: [32.59645168 21.85509089 66.87499616]
    # ----------------------------------------------
    BVHMotion.decompose_rotation_with_yaxis(rotation.as_quat())


test_decompose_rotation_with_yaxis()