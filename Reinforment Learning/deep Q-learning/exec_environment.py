import random
import numpy as np
from zmqRemoteApi import RemoteAPIClient
import tensorflow as tf
from keras import layers, models
import time


class Simulation:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration-exploitation
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.directions = ['Up', 'Down', 'Left', 'Right']
        self.sim_port=23000

        self.client = RemoteAPIClient('localhost', port=23000)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()
        self.td_errors=[]

    
    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    
        model.compile(loss='mse', optimizer=optimizer)
        return model


    def calculate_reward(self, state_matrix):
        reward = 0
        for i in state_matrix:
            if i == 1:
                reward += 1
            else:
                reward += 0
        return reward

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost',port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)  
        
        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()

    def getObjectHandles(self):
        self.tableHandle = self.sim.getObject('/Table')
        self.boxHandle = self.sim.getObject('/Table/Box')

    def dropObjects(self):
        self.blocks = 18
        frictionCube=0.06
        frictionCup=0.8
        blockLength=0.016
        massOfBlock=14.375e-03
        
        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript,self.tableHandle)
        self.client.step()
        retInts,retFloats,retStrings=self.sim.callScriptFunction('setNumberOfBlocks',self.scriptHandle,[self.blocks],[massOfBlock,blockLength,frictionCube,frictionCup],['cylinder'])
        
        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue=self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break

    def getObjectsInBoxHandles(self):
        self.object_shapes_handles = []
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        for obj_handle in self.object_shapes_handles:
            obj_position = self.sim.getObjectPosition(obj_handle, self.sim.handle_world)
            obj_position = np.array(obj_position) - np.array(box_position)
            pos_step.append(list(obj_position[:2]))
        return pos_step


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def replay(self, batch_size,output_file_1="td_errors.txt"):
        
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                current_q_values = self.model.predict(state)[0]
                td_error = target - current_q_values[action]
                self.td_errors.append(td_error)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        self.write_to_file_1(output_file_1,self.td_errors)
    def action(self, direction):
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        _box_position = box_position
        span = 0.02
        steps = 5

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir * span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.client.step()
    def get_state(self):
        quadrants1=[0,0,0,0]
        quadrants2=[0,0,0,0]
        result=[0,0,0,0]
        # box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
        positions = self.getObjectsPositions()
        blue_objs = positions[:9]
        red_objs = positions[9:]

        for blue in blue_objs:
            if blue[0]>0:
                if blue[1]>0:
                    quadrants1[0]+=1
                else:
                    quadrants1[3]+=1
            else :
                if blue[1]>0:
                    quadrants1[1]+=1
                else:
                    quadrants1[2]+=1

        for red in red_objs:
            if red[0]>0:
                if red[1]>0:
                    quadrants2[0]+=1
                else:
                    quadrants2[3]+=1
            else :
                if red[1]>0:
                    quadrants2[1]+=1
                else:
                    quadrants2[2]+=1
        for i in range(4):
            if quadrants1[i]==quadrants2[i] and quadrants1[i]>0 and quadrants2[i]>0:
                result[i]=1
            else:
                result[i]=0
        binary_str = ''.join(map(str, result))

        # Convert binary string to decimal
        decimal_result = int(binary_str, 2)

        return decimal_result,result

    def train(self, episodes, steps,output_file="rewards_file.txt",output1_file="td_errors.txt"):

        rewards=[]
        for episode in range(episodes):
            print(f'Running episode: {episode + 1}')
            state = self.get_state()
            state_1 = np.array(state[1]).reshape(1, -1)
            total_reward = 0

            for step in range(steps):
                action = self.act(state_1)
                self.action(direction=self.directions[action])
                next_state = self.get_state()
                next_state_1 = np.array(next_state[1]).reshape(1, -1)
                reward = self.calculate_reward(next_state[1])
                total_reward += reward
                done = False 
                self.remember(state_1, action, reward, next_state_1, done)
                state = next_state
                if done:
                    break

            print(f'Total reward for episode {episode + 1}: {total_reward}')
            rewards.append(f"episode: {episode +1} - {total_reward}")

        # Replay and calculate TD error
            self.replay(batch_size=32)

            # Decay exploration probability
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            # Save the model periodically
        self.model.save("model.h5")
        self.write_to_file(output_file,rewards)

    def write_to_file(self, output_file, data):
        with open(output_file, 'a') as file:
            for row in data:
              file.write(''.join(map(str, row)) + '\n')
    def write_to_file_1(self, output_file_1, data):
        with open(output_file_1, 'a') as file:
            for td_error_episode in data:
                file.write(f'td error: {td_error_episode}\n')
    
    def test_agent(self,output_file1="test_results.txt"):
        model = models.load_model("model.h5")
        self.model=model
        goal_list=[]
        goal=0
        for episode in range(100):
            print(f'Running episode: {episode + 1}')
            goal_text="unsuccessful"
            state = self.get_state()
            state_1 = np.array(state[1]).reshape(1, -1)

            for step in range(20):
                action = self.act(state_1)
                self.action(direction=self.directions[action])
                next_state = self.get_state()
                next_state_1 = np.array(next_state[1]).reshape(1, -1)
                #goal node
                if next_state[1]==[1,1,1,1]:
                    print(f'Test using agent Episode {episode + 1}: Goal state reached!')
                    goal+=1
                    goal_text=f"successful: {step}"
                    break
            goal_list.append(goal_text)
            

        print("No of times Goal node reached:",goal)

        self.write_to_file(output_file1,goal_list)


def main():
    env = Simulation(state_size=4, action_size=4)
    # env.train(episodes=200, steps=20)
    env.test_agent()
    

if __name__ == '__main__':
    main()
