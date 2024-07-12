import random
from zmqRemoteApi import RemoteAPIClient
import numpy as np

class Simulation():

    def __init__(self, sim_port=23000, episodes=10, steps_per_episode=20, learning_rate=0.1, discount_factor=0.9, exploration_prob=1.0, min_exploration_prob=0.01, exploration_decay=0.995):
        self.sim_port = sim_port
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.exploration_decay = exploration_decay
        self.states = 16
        self.q_table=[]
        for i in range(self.states):
            self.q_table.append([0, 0, 0, 0])
        self.directions = ['Up', 'Down', 'Left', 'Right']

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost', port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')

        # When the simulation is not running, ZMQ message handling could be a bit
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
        frictionCube = 0.06
        frictionCup = 0.8
        blockLength = 0.016
        massOfBlock = 14.375e-03

        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript, self.tableHandle)
        self.client.step()
        retInts, retFloats, retStrings = self.sim.callScriptFunction('setNumberOfBlocks', self.scriptHandle,
                                                                     [self.blocks], [massOfBlock, blockLength, frictionCube, frictionCup], ['cylinder'])

        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue = self.sim.getFloatSignal('toPython')
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
         box_position = self.sim.getObjectPosition(self.boxHandle,self.sim.handle_world)
         for obj_handle in self.object_shapes_handles:
            obj_position = self.sim.getObjectPosition(obj_handle,self.sim.handle_world)
            obj_position = np.array(obj_position) - np.array(box_position)
            pos_step.append(list(obj_position[:2]))
         return pos_step

    def select_action(self, state):
        if self.exploration_prob > 0 and random.uniform(0, 1) < self.exploration_prob:
            action = random.choice(self.directions)
        else:
            action_index = self.q_table[state].index(max(self.q_table[state]))
            action = self.directions[action_index]
        return action

    def update_q_table(self, state, action, reward, next_state):
        max_next_q_value = max(self.q_table[next_state])
        action_index = self.directions.index(action)
        self.q_table[state][action_index] = \
        (1 - self.learning_rate) * self.q_table[state][action_index] + \
        self.learning_rate * (reward + self.discount_factor * max_next_q_value)



    def run_q_learning(self, output_file='q_table_results.txt',output_file1='rewards_for_each_episode.txt'):
        rewards_list=[]
        for episode in range(200):
            self.initializeSim()
            state = self.get_state()
            total_reward = 0

            for step in range(self.steps_per_episode):
                action = self.select_action(state[0])
                self.perform_action(action)
                new_state = self.get_state()
                reward = self.calculate_reward(new_state[1]) 
                self.update_q_table(state[0], action, reward, new_state[0])
                total_reward += reward
                state = new_state

            print(f'Episode {episode + 1}: Total Reward = {total_reward}')
            rewards_list.append(f"episode: {episode +1} - {total_reward}")

            # Decay exploration probability
            self.exploration_prob = max(self.min_exploration_prob, self.exploration_prob * self.exploration_decay)

            self.stopSim()
        self.write_q_table_to_file(output_file,self.q_table)
        self.write_to_file(output_file1,rewards_list)
        

    def write_q_table_to_file(self, output_file, data):
        with open(output_file, 'a') as file:
            for row in data:
              file.write('\t'.join(map(str, row)) + '\n')
    
    def write_to_file(self, output_file, data):
        with open(output_file, 'a') as file:
            for row in data:
              file.write(''.join(map(str, row)) + '\n')
    


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


    def perform_action(self, direction):
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

    def calculate_reward(self, state_matrix):
        reward=0
        for i in state_matrix:
            if i==1:
                reward+=1
            else:
                reward+=0
        return reward


    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()
    
    def load_q_table(self,file_path='q_table_results.txt'):
        with open(file_path,'r') as file:
            lines = file.readlines()
            q_table=[list(map(float, line.strip().split('\t'))) for line in lines]
        return q_table
    def test_q_learning(self,output_file1="random_goalnode_reached.txt",output_file2="Q_table_reached_goalnode.txt"):
        q=self.load_q_table()
        
        goal_list1=[]
        goal=0
        for episode in range(10):
            goal_text1="unsuccessful"
            self.initializeSim()
            state = self.get_state()

            for step in range(self.steps_per_episode):
                action = np.random.choice(self.directions)
                self.perform_action(action)
                new_state = self.get_state()
                state = new_state
                #goal node
                if new_state[1]==[1,1,1,1]:
                    print(f'Test Random Agent Episode {episode + 1}: Goal state reached!')
                    goal+=1
                    goal_text1=f"successful: {step}"
                    break
            goal_list1.append(goal_text1)
            self.stopSim()
                    
        print("No of times Goal node reached:",goal)
        self.write_to_file(output_file1,goal_list1)
        goal_list=[]
        goal=0
        for episode in range(10):
            goal_text="unsuccessful"  
            self.initializeSim()
            state = self.get_state()

            for step in range(self.steps_per_episode):
                action_index = np.argmax(self.q_table[state[0]])
                action = self.directions[action_index]
                self.perform_action(action)
                new_state = self.get_state()
                state = new_state
                #goal node
                if new_state[1]==[1,1,1,1]:
                    print(f'Test using Q Table Episode {episode + 1}: Goal state reached!')
                    goal+=1
                    goal_text=f"successful: {step}"
                    break
            goal_list.append(goal_text)
            self.stopSim()

        print("No of times Goal node reached:",goal)
        
        self.write_to_file(output_file2,goal_list)


def main():
    env = Simulation()
    env.run_q_learning()
    env.test_q_learning()

if __name__ == '__main__':
    main()
