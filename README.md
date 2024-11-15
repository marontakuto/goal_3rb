# multi_avoidance

Gazeboで群ロボットの回避行動の獲得を実現するためのリポジトリです。


## 環境構築
1. [こちらのサイト](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/)へアクセス
2. [3. Quick Start Guide]3.1.1~3.1.4のコマンドを順に実行する(3.1.4の「Click here to expand more details about building TurtleBot3 package from source.」も実行する)
3. [9. Machine Learning]上部の「Kinetic」「Melodic」「Noetic」... より「Melodic」を選択し、9.1.5のコマンドを順番に入力する(numpyのコマンドは無視)
4. [6. Simulation]6.1.2のコマンドを実行してワールドとturtlebot3 burgerが出力されればOK
5. cd ~/catkin_ws/turtlebot3_learning_machine/turtlebot3_dqn
6. git clone https://github.com/marontakuto/multi_avoidance.git
7. ターミナルを2つ開く
8. roslaunch turtlebot3_dqn lab_goal_multi.launch # 1つ目のターミナルで実行
9. roslaunch turtlebot3_dqn turtlebot3_3robot_avoid.launch # 2つ目のターミナルで実行