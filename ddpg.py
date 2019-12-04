from gym_torcs import TorcsEnv
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.engine.training import collect_trainable_weights
import json
import csv
import datetime
import os

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

OU = OU()       # Ornstein-Uhlenbeck Process


def playGame(train_indicator=1):    # 1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000  # 缓存能力，网络储存能力
    BATCH_SIZE = 32  # 批尺寸，一次处理样本数
    GAMMA = 0.99  # 折扣系数
    TAU = 0.001     # Target Network HyperParameters 目标网络超系数
    LRA = 0.0001    # Learning rate for Actor Actor网络学习率
    LRC = 0.001     # Lerning rate for Critic Critic网络学习率

    action_dim = 3  # Steering/Acceleration/Brake 加速/转向/刹车
    state_dim = 29  # of sensors input 29个传感器输入

    np.random.seed(1337)  # 随机数种子，如果使用相同的数字，则每次产生的随机数相同，应该是定义了一个随机的初始值。

    vision = False

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    # Tensorflow GPU 管理策略，此处使用动态内存申请策略
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # 硬性限制GPU使用率为0.4
    # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    # Create replay buffer

    #  Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True,gear_change=False)

    # Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    theTime = datetime.datetime.now()  # 获取系统当前时间
    theTime = theTime.strftime('%y-%m-%d_%H:%M:%S')  # 转换为字符串形式作为CSV文件头
    folder_path = "practise_progress/" + theTime + "/"  # 只适用于Linux系统
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("folder created")
    else:
        print("folder existed")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        if np.mod(i, 3) == 0:
            ob = env.reset(relaunch=True)   # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
     
        total_reward = 0.

        csvfileHeader = "practise_progress/" + theTime + "/" + " Episode " + str(i) + ".csv"
        fileHeader = ["Step", "TrackPos", "SpeedX", "SpeedY", "SpeedZ",
                      "Action_Steering", "Action_Acceleration", "Action_Brake", "Reward", "Loss"]
        csvFile = open(csvfileHeader, "w")
        writer = csv.writer(csvFile)
        writer.writerow(fileHeader)

        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])
            
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0],  0.0 , 0.60, 0.30)
            noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1],  0.5 , 1.00, 0.10)
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1 , 1.00, 0.05)

            # The following code do the stochastic brake
            # if random.random() <= 0.1:
            #     print("********Now we apply the brake***********")
            #     noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2],  0.2 , 1.00, 0.10)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

            ob, r_t, done, info = env.step(a_t[0])

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        
            buff.add(s_t, a_t[0], r_t, s_t1, done)      # Add replay buffer
            
            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if (train_indicator):
                loss += critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            csvData = [step, ob.trackPos, ob.speedX * 300, ob.speedY * 300, ob.speedZ * 300,
                       a_t[0, 0], a_t[0, 1], a_t[0, 2], r_t, loss]
            """        参数记录
                       轮次  步骤计数  车辆位置  X轴速度  Y轴速度  Z轴速度
                       加速输出  转向输出  刹车输出  回报  损失函"""
            writer.writerow(csvData)
            print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
            step += 1
            if done:
                csvFile.close()
                break

        if np.mod(i, 3) == 0:
            if (train_indicator):
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)



        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  #  This is for shutting down TORCS
    print("Finish.")

if __name__ == "__main__":
    playGame()
