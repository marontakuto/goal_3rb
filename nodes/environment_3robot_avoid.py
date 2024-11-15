#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import random
import math
import time
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String, Float32, Float32MultiArray, UInt16, Int32
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from collections import deque
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from tf.transformations import *
import itertools

import cv2
from sensor_msgs.msg import Image, CompressedImage
import sys
import ros_numpy

class Env():
    def __init__(self, mode, robot_n, lidar_num, input_list, r_collision,  r_just, r_near, r_goal, Target):
        
        self.mode = mode
        self.robot_n = robot_n
        self.lidar_num = lidar_num
        self.input_list = input_list
        self.previous_cam_list = deque([])
        self.previous_lidar_list = deque([])
        self.previous2_cam_list = deque([])
        self.previous2_lidar_list = deque([])
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

        # 画像取得の前処理
        if ('cam'in self.input_list) or ('previous_cam'in self.input_list) or ('previous2_cam'in self.input_list): 
            if self.mode == 'sim':
                self.sub_img = rospy.Subscriber('usb_cam/image_raw', Image, self.pass_img, queue_size=10) # シミュレーション用
            else:
                self.sub_img = rospy.Subscriber('usb_cam/image_raw/compressed', CompressedImage, self.pass_img, queue_size=10) # 実機用

        self.lidar_max = 2 # 対象のworldにおいて取りうるlidarの最大値(simの貫通対策や正規化に使用)
        self.lidar_min = 0.12 # lidarの最小測距値[m]
        self.range_margin = self.lidar_min + 0.03 # 衝突として処理される距離[m] 0.02
        self.display_image = False # 入力画像を表示する
        self.start_time = self.get_clock() # トライアル開始時の時間取得

        # Optunaで選択された報酬値
        self.r_collision = r_collision
        self.r_just = r_just
        self.r_near = r_near
        self.r_goal = r_goal
        self.Target = Target

    def pass_img(self,img): # 画像正常取得用callback
        pass

    def get_clock(self): # シミュレーションでの倍速に対して当倍速として時間を取得する
        if self.mode == 'sim':
            data = None
            while data is None:
                try:
                    data = rospy.wait_for_message('clock', Clock, timeout=10)
                except:
                    time.sleep(2)
                    print('Please check "mode"!')
                    pass
            secs = data.clock.secs
            nsecs = data.clock.nsecs / 10 ** 9
            return secs + nsecs
        else:
            return time.time()
        
    def get_lidar(self, restart=False, retake=False): # lidar情報の取得
        if retake:
            self.scan = None
        
        if self.scan is None:

            scan = None
            while scan is None:
                try:
                    scan = rospy.wait_for_message('scan', LaserScan, timeout=1) # LiDAR値の取得(1deg刻み360方向の距離情報を取得)
                except:
                    self.stop()
                    pass
            
            data_range = [] # 取得したLiDAR値を修正して格納するリスト
            for i in range(len(scan.ranges)):
                if scan.ranges[i] == float('Inf'): # 最大より遠いなら3.5
                    data_range.append(3.5)
                if np.isnan(scan.ranges[i]): # 最小より近いなら0
                    data_range.append(0)
                
                if self.mode == 'sim':
                    if scan.ranges[i] > self.lidar_max: # フィールドで観測できるLiDAR値を超えていたら0
                        data_range.append(0)
                    else:
                        data_range.append(scan.ranges[i]) # 取得した値をそのまま利用
                else:
                    data_range.append(scan.ranges[i]) # 実機では取得した値をそのまま利用

            use_list = [] # 計算に利用するLiDAR値を格納するリスト
            if restart:
                # lidar値を45deg刻み8方向で取得(リスタート用)
                for i in range(8):
                    index = (len(data_range) // 8) * i
                    scan = max(data_range[index - 2], data_range[index - 1], data_range[index], data_range[index + 1], data_range[index + 2]) # 実機の飛び値対策(値を取得できず0になる場合があるため前後2度で最大の値を採用)
                    use_list.append(scan)
            else:
                # lidar値を[360/(self.lidar_num)]deg刻み[self.lidar_num]方向で取得
                for i in range(self.lidar_num):
                    index = (len(data_range) // self.lidar_num) * i
                    scan = max(data_range[index - 2], data_range[index - 1], data_range[index], data_range[index + 1], data_range[index + 2]) # 実機の飛び値対策(値を取得できず0になる場合があるため前後2度で最大の値を採用)
                    use_list.append(scan)
            
            self.scan = use_list

        return self.scan

    def get_camera(self, retake=False): # camera画像取得

        if retake:
            self.img = None
        
        if self.img is None:
            img = None
            while img is None:
                try:
                    if self.mode == 'sim':
                       img = rospy.wait_for_message('usb_cam/image_raw', Image, timeout=1) # シミュレーション用(生データ)
                    else:
                       img = rospy.wait_for_message('usb_cam/image_raw/compressed', CompressedImage, timeout=1) # 実機用(圧縮データ)
                except:
                    self.stop()
                    pass

            if self.mode == 'sim':
                img = ros_numpy.numpify(img)
            else:
                img = np.frombuffer(img.data, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # カラー画像
            
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # グレースケール

            img = cv2.resize(img, (48, 27)) # 取得した画像を48×27[pixel]に変更

            if self.display_image:
                disp_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 標準フォーマットBGR
                cv2.namedWindow('camera', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('camera', 480, 270) # 480×270[pixel]のウィンドウで表示
                cv2.imshow('camera', disp_img) # 表示
                cv2.waitKey(1)
            
            self.img = img

        return self.img

    def get_count(self, img, middle=False, left=False, middle2=False, all=False, middle3=False, right=False, left2=False, left3=False, left4=False, left5=False, side=False):
        # 色の設定
        outside_lower = np.array([0, 0, 0], dtype=np.uint8) # 黒
        outside_upper = np.array([255, 255, 80], dtype=np.uint8)
        inside_lower = np.array([0, 150, 90], dtype=np.uint8) # オレンジ
        inside_upper = np.array([20, 255, 255], dtype=np.uint8) 
        robot_blue_lower = np.array([85, 100, 50], dtype=np.uint8) # ブルー
        robot_blue_upper = np.array([130, 255, 255], dtype=np.uint8)
        robot_green_lower = np.array([55, 100, 60], dtype=np.uint8) # グリーンnp.array([40, 50, 50])
        robot_green_upper = np.array([85, 255, 255], dtype=np.uint8)

        if middle:
            img = img[:, len(img[0]) * 4 // 9:len(img[0]) * 5 // 9]
        if middle2:
            img = img[:, len(img[0]) * 1 // 3:len(img[0]) * 2 // 3] # 1/3 ~ 2/3
        if middle3:
            img = img[:, len(img[0]) * 1 // 5:len(img[0]) * 4 // 5]
        if left:
            img = img[:, 0:len(img[0]) * 1 // 9]
        if left2:
            img = img[:, 0:len(img[0]) * 4 // 9]
        if left3:
            img = img[:, 0:len(img[0]) * 1 // 2]
        if left4:
            img = img[:, 0:len(img[0]) * 1 // 3]
        if left5:
            img = img[:, 0:len(img[0]) * 1 // 5]
        if right:
            img = img[:, len(img[0]) * 8 // 9:len(img[0]) * 9 // 9]
        if all:
            img = img[:, len(img[0]) * 0 // 9:len(img[0]) * 9 // 9]
        if side:
            img = np.hstack((img[:, :len(img[1]) * 1 // 5], img[:, len(img[1]) * 4 // 5:]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        outside = cv2.inRange(img_hsv, outside_lower, outside_upper)
        inside = cv2.inRange(img_hsv, inside_lower, inside_upper)
        robot = cv2.inRange(img_hsv, robot_blue_lower, robot_blue_upper)
        robot2 = cv2.inRange(img_hsv, robot_green_lower, robot_green_upper)

        outside_num = np.count_nonzero(outside)
        inside_num = np.count_nonzero(inside)
        robot_blue_num = np.count_nonzero(robot)
        robot_green_num = np.count_nonzero(robot2)

        if self.display_image:
            disp_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 標準フォーマットBGR
            disp_img = np.hstack((disp_img[:, :len(img[1]) * 1 // 5], disp_img[:, len(img[1]) * 4 // 5:]))
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('img', 480, 270)
            cv2.imshow('img', disp_img)
            cv2.waitKey(1)
        
        return outside_num, inside_num, robot_blue_num, robot_green_num

    def getState(self): # 情報取得

        state_list = [] # 入力する情報を格納するリスト
        img = self.get_camera() # カメラ画像の取得
        scan = self.get_lidar() # LiDAR値の取得
        collision = False

        # 入力するカメラ画像の処理
        if ('cam' in self.input_list) or ('previous_cam' in self.input_list) or ('previous2_cam' in self.input_list):
            input_img = np.asarray(img, dtype=np.float32)
            input_img /= 255.0 # 画像の各ピクセルを255で割ることで0~1の値に正規化
            input_img = np.asarray(input_img.flatten())
            input_img = input_img.tolist()

            state_list = state_list + input_img # [現在]

            if 'previous_cam' in self.input_list:
                self.previous_cam_list.append(input_img)
                if len(self.previous_cam_list) > 2: # [1step前 現在]の状態に保つ
                    self.previous_cam_list.popleft() # 左端の要素を削除(2step前の情報を削除)
                past_cam = self.previous_cam_list[0] # 1step前の画像
                state_list = state_list + past_cam # [現在 1step前]

                if 'previous2_cam' in self.input_list:
                    self.previous2_cam_list.append(input_img)
                    if len(self.previous2_cam_list) > 3: # [2step前 1step前 現在]の状態に保つ
                        self.previous2_cam_list.popleft() # 左端の要素を削除(3step前の情報を削除)
                    past_cam = self.previous2_cam_list[0] # 2step前の画像
                    state_list = state_list + past_cam # [現在 1step前 2step前]

        # 入力するLiDAR値の処理
        if ('lidar' in self.input_list) or ('previous_lidar' in self.input_list) or ('previous2_lidar' in self.input_list):
            input_scan = [] # 正規化したLiDAR値を格納するリスト
            for i in range(len(scan)): # lidar値の正規化
                input_scan.append((scan[i] - self.range_margin) / (self.lidar_max - self.range_margin))

            state_list = state_list + input_scan # [画像] + [現在]

            if 'previous_lidar' in self.input_list:
                self.previous_lidar_list.append(input_scan)
                if len(self.previous_lidar_list) > 2: # [1step前 現在]の状態に保つ
                    self.previous_lidar_list.popleft() # 左端の要素を削除(2step前の情報を削除)
                past_scan = self.previous_lidar_list[0] # 1step前のLiDAR値
                state_list = state_list + past_scan # [画像] + [現在 1step前]

                if 'previous2_lidar' in self.input_list:
                    self.previous2_lidar_list.append(input_scan)
                    if len(self.previous2_lidar_list) > 3: # [2step前 1step前 現在]の状態に保つ
                        self.previous2_lidar_list.popleft() # 左端の要素を削除(3step前の情報を削除)
                    past_scan = self.previous2_lidar_list[0] # 2step前のLiDAR値
                    state_list = state_list + past_scan # [画像] + [現在 1step前 2step前]
        
        # LiDAR値による衝突判定
        if self.range_margin >= min(scan):
            collision = True
            if self.mode == 'real': # 実機実験におけるLiDARの飛び値の処理
                scan_true = [element_cont for element_num, element_cont in enumerate(scan) if element_cont != 0]
                if scan.count(0) >= 1 and self.range_margin < min(scan_true): # (飛び値が存在する)and(飛び値を除いた場合は衝突判定ではない)
                    collision = False
        
        # 画像情報によるゴール判定
        goal, goal_num = self.goal_judge(img)
        
        return state_list, img, scan, input_scan, collision, goal, goal_num
   
    def setReward(self, img, scan, collision, goal, goal_num, action):

        reward = 0
        color_num = 0
        just_count = 0

        # _, _, _, _ = self.get_count(img, all=True)
        color_num = goal_num

        if self.Target == 'both' or self.Target == 'reward':
            if goal:
                reward += self.r_goal
                just_count = 1
            elif collision:
                reward -= self.r_collision
            reward += goal_num * self.r_just
            reward -= min(1 / (min(scan) + 0.01), 7) * self.r_near
        else:
            if goal:
                reward += 50 # r_goal
                just_count = 1
            elif collision:
                reward -= 50 # r_collision
            reward += goal_num * 1 # r_just
            reward -= min(1 / (min(scan) + 0.01), 7) * 1 # r_near
        
        return reward, color_num, just_count

    def step(self, action, deceleration, teleport, test): # 1stepの行動

        self.img = None
        self.scan = None

        vel_cmd = Twist()

        "最大速度 x: 0.22[m/s], z: 2.84[rad/s](162.72[deg/s])"
        "z値 0.785375[rad/s] = 45[deg/s], 1.57075[rad/s] = 90[deg/s], 2.356125[rad/s] = 135[deg/s]"
        "行動時間は行動を決定してから次の行動が決まるまでのため1秒もない"

        if action == 0: # 左折
            vel_cmd.linear.x = 0.2 # 直進方向[m/s]
            vel_cmd.angular.z = 1.57 # 回転方向 [rad/s]
        
        elif action == 1: # 直進
            vel_cmd.linear.x = 0.15 # 直進方向[m/s]
            vel_cmd.angular.z = 0 # 回転方向[rad/s]

        elif action == 2: # 右折
            vel_cmd.linear.x = 0.2 # 直進方向[m/s]
            vel_cmd.angular.z = -1.57 # 回転方向[rad/s]
        
        elif action == 3: # 直進(低速)
            vel_cmd.linear.x = 0.1 # 直進方向[m/s]
            vel_cmd.angular.z = 0 # 回転方向[rad/s]
                
        if action == 99: # 停止
            vel_cmd.linear.x = 0 # 直進方向[m/s]
            vel_cmd.angular.z = 0 # 回転方向[rad/s]
        
        vel_cmd.linear.x = vel_cmd.linear.x * deceleration
        vel_cmd.linear.z = vel_cmd.linear.z * deceleration
        
        self.pub_cmd_vel.publish(vel_cmd) # 実行
        state_list, img, scan, input_scan, collision, goal, goal_num = self.getState()
        reward, color_num, just_count = self.setReward(img, scan, collision, goal, goal_num, action)

        if (collision or goal) and not test: # 衝突時の処理(テスト時を除く)
            if not teleport:
                self.restart() # 進行方向への向き直し
            elif teleport:
                self.relocation() # 空いているエリアへの再配置
                time.sleep(0.1)
        
        return np.array(state_list), reward, color_num, just_count, collision, goal, input_scan

    def reset(self):
        self.img = None
        self.scan = None
        state_list, _, _, _, _, _, _ = self.getState()
        return np.array(state_list)
    
    def restart(self):

        self.stop()
        vel_cmd = Twist()

        data_range = self.get_lidar(restart=True, retake=True)

        while True:
            while True:
                
                vel_cmd.linear.x = 0 # 直進方向[m/s]
                vel_cmd.angular.z = pi/4 # 回転方向[rad/s]
                self.pub_cmd_vel.publish(vel_cmd) # 実行
                data_range = self.get_lidar(restart=True, retake=True)
                
                if data_range.index(min(data_range)) == 0: # 正面
                    wall = 'front'
                    break
                if data_range.index(min(data_range)) == round(len(data_range)/2): # 背面
                    wall = 'back'
                    break

                self.stop()

            if wall =='front':
                while data_range[0] < self.range_margin + 0.10: # 衝突値＋10cm
                    vel_cmd.linear.x = -0.10 # 直進方向[m/s]
                    vel_cmd.angular.z = 0  # 回転方向[rad/s]
                    self.pub_cmd_vel.publish(vel_cmd) # 実行
                    data_range = self.get_lidar(restart=True, retake=True)

            elif wall =='back':
                while data_range[round(len(data_range)/2)] < self.range_margin + 0.10: # 衝突値＋10cm
                    vel_cmd.linear.x = 0.10 # 直進方向[m/s]
                    vel_cmd.angular.z = 0  # 回転方向[rad/s]
                    self.pub_cmd_vel.publish(vel_cmd) # 実行
                    data_range = self.get_lidar(restart=True,retake=True)
            
            self.stop()
            
            side_list = [round(len(data_range) / 4), round(len(data_range) * 3 / 4)] # 側面
            num = np.random.randint(0, 2)
            while True:
                vel_cmd.linear.x = 0 # 直進方向[m/s]
                if num == 0:
                    vel_cmd.angular.z = pi/4 # 回転方向[rad/s]
                else:
                    vel_cmd.angular.z = -pi/4 # 回転方向[rad/s]
                self.pub_cmd_vel.publish(vel_cmd) # 実行
                data_range = self.get_lidar(restart=True, retake=True)
                if data_range.index(min(data_range)) in side_list: # 側面のLiDAR値が最小である時
                    break
            
            self.stop()

            data_range = self.get_lidar(restart=True, retake=True)
            if min(data_range) > self.range_margin + 0.05: # LiDAR値が衝突判定の距離より余裕がある時
                break
    
    def set_robot(self, num): # 指定位置にロボットを配置

        self.stop()
        
        a = [0.55, 0.35, 0.02, 2.355] # 右上
        b = [0.55, 1.45, 0.02, -2.355] # 左上
        c = [-0.55, 1.45, 0.02, -0.785] # 左下

        if num == 0: # 初期位置
            if self.robot_n == 0:
                XYZyaw = a
            if self.robot_n == 1:
                XYZyaw = b
            if self.robot_n == 2:
                XYZyaw = c
              
        elif num == 99: # フィールド外
            if self.robot_n == 0:
                XYZyaw = [0, -0.5, 0.02, 3.14] # 右
            if self.robot_n == 1:
                XYZyaw = [1.4, 0.9, 0.02, -1.57] # 上
            if self.robot_n == 2:
                XYZyaw = [0, 2.2, 0.02, 0.0] # 左
        
        elif num == 101: # フィールドの中心
            if self.robot_n == 0:
                XYZyaw = [-0.04, 0.77, 0.27, 0] # 右
            if self.robot_n == 1:
                XYZyaw = [0.13, 0.92, 0.27, 3.14] # 上
            if self.robot_n == 2:
                XYZyaw = [-0.13, 1.02, 0.27, 0] # 左
        
        elif num == 102: # フィールド外の右側
            if self.robot_n == 0:
                XYZyaw = [-0.55, -0.3, 0.02, 0] # 下
            if self.robot_n == 1:
                XYZyaw = [0.0, -0.3, 0.02, 3.14] # 中央
            if self.robot_n == 2:
                XYZyaw = [0.55, -0.3, 0.02, 0] # 上
        
        elif num == 103: # フィールド外の左側
            if self.robot_n == 0:
                XYZyaw = [0.55, 2.1, 0.02, 0] # 下
            if self.robot_n == 1:
                XYZyaw = [0.0, 2.1, 0.02, 3.14] # 中央
            if self.robot_n == 2:
                XYZyaw = [-0.55, 2.1, 0.02, 0] # 上
        
        elif num == 104: # フィールド外の下側
            if self.robot_n == 0:
                XYZyaw = [-1.2, 1.45, 0.02, 0] # 左
            if self.robot_n == 1:
                XYZyaw = [-1.2, 0.9, 0.02, 3.14] # 中央
            if self.robot_n == 2:
                XYZyaw = [-1.2, 0.35, 0.02, 0] # 右
        
        elif num == 105: # フィールド外の上側
            if self.robot_n == 0:
                XYZyaw = [1.2, 0.35, 0.02, 0] # 右
            if self.robot_n == 1:
                XYZyaw = [1.2, 0.9, 0.02, 3.14] # 中央
            if self.robot_n == 2:
                XYZyaw = [1.2, 1.45, 0.02, 0] # 左
        

        # 空いたエリアへのロボットの配置用[relocation()]
        if num == 11:
            XYZyaw = [0.55, 0.9, 0.02, 3.14] # 上
        elif num == 12:
            XYZyaw = [0.55, 0.35, 0.02, 2.355] # 右上
        elif num == 13:
            XYZyaw = [0.0, 0.35, 0.02, 1.57] # 右
        elif num == 14:
            XYZyaw = [-0.55, 0.35, 0.02, 0.785] # 右下
        elif num == 15:
            XYZyaw = [-0.55, 0.9, 0.02, 0.0] # 下
        elif num == 16:
            XYZyaw = [-0.55, 1.45, 0.02, -0.785] # 左下
        elif num == 17:
            XYZyaw = [0.0, 1.45, 0.02, -1.57] # 左
        elif num == 18:
            XYZyaw = [0.55, 1.45, 0.02, -2.355] # 左上


        if num == 1: # 以下テスト用
            if self.robot_n == 0:
                XYZyaw = a
            if self.robot_n == 1:
                XYZyaw = b
            if self.robot_n == 2:
                XYZyaw = c
        
        elif num == 2:
            if self.robot_n == 0:
                XYZyaw = a
            if self.robot_n == 1:
                XYZyaw = b
            if self.robot_n == 2:
                XYZyaw = c

        elif num == 3:
            if self.robot_n == 0:
                XYZyaw = a
            if self.robot_n == 1:
                XYZyaw = b
            if self.robot_n == 2:
                XYZyaw = c

        elif num == 4:
            if self.robot_n == 0:
                XYZyaw = a
            if self.robot_n == 1:
                XYZyaw = b
            if self.robot_n == 2:
                XYZyaw = c

        # elif num == 2:
        #     if self.robot_n == 0:
        #         XYZyaw = a
        #     if self.robot_n == 1:
        #         XYZyaw = [0.0, 1.45, 0.02, -1.57] # 左
        #     if self.robot_n == 2:
        #         XYZyaw = c

        # elif num == 3:
        #     if self.robot_n == 0:
        #         XYZyaw = a
        #     if self.robot_n == 1:
        #         XYZyaw = [0.55, 0.9, 0.02, 3.14] # 上
        #     if self.robot_n == 2:
        #         XYZyaw = c

        # elif num == 4:
        #     if self.robot_n == 0:
        #         XYZyaw = a
        #     if self.robot_n == 1:
        #         XYZyaw = [-0.55, 0.35, 0.02, 0.785] # 右下
        #     if self.robot_n == 2:
        #         XYZyaw = c
        
        state_msg = ModelState()
        state_msg.model_name = 'tb3_{}'.format(self.robot_n)

        state_msg.pose.position.x = XYZyaw[0]
        state_msg.pose.position.y = XYZyaw[1]
        state_msg.pose.position.z = XYZyaw[2]
        q = quaternion_from_euler(0, 0, XYZyaw[3])
        state_msg.pose.orientation.x = q[0]
        state_msg.pose.orientation.y = q[1]
        state_msg.pose.orientation.z = q[2]
        state_msg.pose.orientation.w = q[3]
        rospy.wait_for_service('/gazebo/set_model_state')
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        set_state(state_msg)

    # 以降追加システム

    def goal_judge(self, img): # ロボットのゴール判定

        goal = False

        if self.robot_n == 0: # オレンジ
            goal_lower = np.array([0, 150, 90])
            goal_upper = np.array([20, 255, 255])
        elif self.robot_n == 1: # 緑
            goal_lower = np.array([55, 100, 60])
            goal_upper = np.array([85, 255, 255])
        elif self.robot_n == 2: # 黄
            goal_lower = np.array([15, 150, 90])
            goal_upper = np.array([35, 255, 255])

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #HSV
        goal_mask = cv2.inRange(img_hsv, goal_lower, goal_upper)
        goal_num = np.count_nonzero(goal_mask)
        if goal_num > 300:
            goal = True
        
        return goal, goal_num

    def stop(self): # ロボットの停止
        vel_cmd = Twist()
        vel_cmd.linear.x = 0 # 直進方向[m/s]
        vel_cmd.angular.z = 0  # 回転方向[rad/s]
        self.pub_cmd_vel.publish(vel_cmd) # 実行
    
    def robot_coordinate(self): # ロボットの座標を取得
        ros_data = None
        while ros_data is None:
            try:
                ros_data = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=1) # ROSデータの取得
            except:
                pass
        
        tb3_0 = ros_data.name.index('tb3_0') # robot0のデータの配列番号
        tb3_1 = ros_data.name.index('tb3_1') # robot1のデータの配列番号
        tb3_2 = ros_data.name.index('tb3_2') # robot2のデータの配列番号

        rb0 = np.array([ros_data.pose[tb3_0].position.x, ros_data.pose[tb3_0].position.y], dtype='float') # robot0の座標
        rb1 = np.array([ros_data.pose[tb3_1].position.x, ros_data.pose[tb3_1].position.y], dtype='float') # robot1の座標
        rb2 = np.array([ros_data.pose[tb3_2].position.x, ros_data.pose[tb3_2].position.y], dtype='float') # robot2の座標

        return rb0, rb1, rb2

    def relocation(self): # 衝突時、ほかロボットの座標を観測し、空いている座標へ配置

        exist_erea = [] # ロボットが存在するエリアを格納するリスト

        # 各ロボットの座標
        rb0, rb1, rb2 = self.robot_coordinate()

        if self.robot_n == 0:
            coordinate_list = [rb1, rb2]
        elif self.robot_n == 1:
            coordinate_list = [rb0, rb2]
        elif self.robot_n == 2:
            coordinate_list = [rb0, rb1]

        for coordinate in coordinate_list:
            if 0.3 <= coordinate[0] <= 0.9 and 0.6 <= coordinate[1] <= 1.2: # 上のエリアに存在するか
                exist_erea.append(1)
            elif 0.3 <= coordinate[0] <= 0.9 and 0.0 <= coordinate[1] <= 0.6: # 右上のエリアに存在するか
                exist_erea.append(2)
            elif -0.3 <= coordinate[0] <= 0.3 and 0.0 <= coordinate[1] <= 0.6: # 右のエリアに存在するか
                exist_erea.append(3)
            elif -0.9 <= coordinate[0] <= -0.3 and 0.0 <= coordinate[1] <= 0.6: # 右下のエリアに存在するか
                exist_erea.append(4)
            elif -0.9 <= coordinate[0] <= -0.3 and 0.6 <= coordinate[1] <= 1.2: # 下のエリアに存在するか
                exist_erea.append(5)
            elif -0.9 <= coordinate[0] <= -0.3 and 1.2 <= coordinate[1] <= 1.8: # 左下のエリアに存在するか
                exist_erea.append(6)
            elif -0.3 <= coordinate[0] <= 0.3 and 1.2 <= coordinate[1] <= 1.8: # 左のエリアに存在するか
                exist_erea.append(7)
            elif 0.3 <= coordinate[0] <= 0.9 and 1.2 <= coordinate[1] <= 1.8: # 左上のエリアに存在するか
                exist_erea.append(8)
        
        # 空いているエリア
        empty_area = [x for x in list(range(8, 0, -1)) if x not in exist_erea]
        
        # テレポートさせるエリア
        if self.robot_n == 0:
            teleport_area = empty_area[-1]
            if 2 in empty_area:
                teleport_area = 2
        elif self.robot_n == 1:
            teleport_area = empty_area[0]
            if 8 in empty_area:
                teleport_area = 8
        elif self.robot_n == 2:
            teleport_area = empty_area[2]
            if 6 in empty_area:
                teleport_area = 6
        
        # テレポート
        self.set_robot(teleport_area + 10)

    def area_judge(self, terms, area):
        exist = False
        judge_list = []
        rb0, rb1, rb2 = self.robot_coordinate() # ロボットの座標を取得

        # エリアの座標を定義
        if area == 'right':
            area_coordinate = [-0.9, 0.9, -1.8, 0.0] # [x_最小, x_最大, y_最小, y_最大]
        elif area == 'left':
            area_coordinate = [-0.9, 0.9, 1.8, 3.6]
        elif area == 'lower':
            area_coordinate = [-2.7, -0.9, 0.0, 1.8]
        elif area == 'upper':
            area_coordinate = [0.9, 2.7, 0.0, 1.8]
        
        # 他のロボットの座標を格納
        if self.robot_n == 0:
            judge_robot = [rb1, rb2]
        elif self.robot_n == 1:
            judge_robot = [rb0, rb2]
        elif self.robot_n == 2:
            judge_robot = [rb0, rb1]
        
        # 他のロボットのエリア内外判定
        for rb in judge_robot:
            judge_list.append(area_coordinate[0] < rb[0] < area_coordinate[1] and area_coordinate[2] < rb[1] < area_coordinate[3])
        
        if terms == 'hard' and (judge_list[0] and judge_list[1]): # 他の全ロボットがエリアに存在する時
            exist = True
        elif terms == 'hsoft' and (judge_list[0] or judge_list[1]): # 他のロボットが1台でもエリアに存在する時
            exist = True

        return exist

    # 以降リカバリー方策
    def recovery_deceleration(self, input_scan, lidar_num): # LiDAR前方の数値が低い場合は減速させる

        threshold = 0.1 # 減速させ始める距離
        forward = list(range(round(lidar_num*0.9)+1, lidar_num)) + list(range(0, round(lidar_num*0.1))) # LiDARの前方とする要素番号(左右30度ずつ)

        # LiDARのリストで条件に合う要素を格納したリストをインスタンス化(element_num:要素番号, element_cont:要素内容)
        low_lidar = [element_num for element_num, element_cont in enumerate(input_scan) if element_cont <= threshold]

        # 指定したリストと条件に合う要素のリストで同じ数字があった場合
        if set(forward) & set(low_lidar) != set():
            deceleration = 0.7 # 元の速度の何%の速度にするか
            # print(self.robot_n)
        else:
            deceleration = 1 # 減速なし
        
        return deceleration
    
    def recovery_change_action(self, e, input_scan, lidar_num, action, state, model): # LiDARの数値が低い方向への行動を避ける

        ### ユーザー設定パラメータ ###
        threshold = 0.18 # 何mでセンサーが反応したら動きを変えるか決める数値
        probabilistic = True # True: リカバリー方策を確率的に利用する, False: リカバリー方策を必ず利用する
        initial_probability = 1.0 # 最初の確率
        finish_episode = 15 # 方策を適応する最後のエピソード
        ##############################

        # リカバリー方策の利用判定
        if not probabilistic: # 必ず利用
            pass
        elif random.random() < round(initial_probability - (initial_probability / finish_episode) * (e - 1), 3): # 確率で利用(確率は線形減少)
            pass
        else:
            return action
        
        change_action = False
        bad_action = []

        # 方向の定義
        left = list(range(2, lidar_num // 4 + 1)) # LiDARの前方左側
        forward = [0, 1, lidar_num - 1] # LiDARの前方とする要素番号(左右数度ずつ)
        right = list(range(lidar_num * 3 // 4, lidar_num - 1)) # LiDARの前方右側

        # LiDARのリストで条件に合う要素を格納したリストをインスタンス化(element_num:要素番号, element_cont:要素内容)
        low_lidar = [element_num for element_num, element_cont in enumerate(input_scan) if element_cont <= threshold]

        # 指定したリストと条件に合う要素のリストで同じ数字があった場合は行動を変更する(actionを 0は左折, 1は直進, 2は右折 に設定する必要あり)
        if set(left) & set(low_lidar) != set():
            bad_action.append(0)
            if action == 0:
                change_action = True
        if set(forward) & set(low_lidar) != set():
            bad_action.append(1)
            if action == 1 or action == 3:
                change_action = True
        if set(right) & set(low_lidar) != set():
            bad_action.append(2)
            if action == 2:
                change_action = True
        
        # 行動を変更
        if change_action:
            net_out = model.forward(state.unsqueeze(0).to('cuda:0')) # ネットワークの出力
            q_values = net_out.q_values.cpu().detach().numpy().tolist()[0] # Q値

            if len(bad_action) == 3: # 全方向のLiDAR値が低い場合はLiDAR値が最大の方向へ
                front = input_scan[0:10] + input_scan[27:36] # left~forward~rightまで
                max_index = input_scan.index(max(front)) # left~forward~rightまでで最大の値の要素番号
                if max_index in left: # left方向が空いている場合は左折
                    action = 0
                elif max_index in forward: # forward方向が空いている場合は直進
                    action = 1
                elif max_index in right: # right方向が空いている場合は右折
                    action = 2
            elif len(bad_action) == 2: # 2方向のLiDAR値が低い場合は残りの方向へ
                action = (set([0, 1, 2]) - set(bad_action)).pop()
            elif len(bad_action) == 1: # 1方向のLiDAR値が低い場合はQ値が大きい方向へ
                action_candidate = list(set([0, 1, 2]) - set(bad_action))
                if q_values[action_candidate[0]] > q_values[action_candidate[1]]:
                    action = action_candidate[0]
                else:
                    action = action_candidate[1]

        return action