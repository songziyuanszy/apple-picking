import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

"""     两警察围剿一小偷，又名 “二龙戏珠”     """
"""action=[policeA的速度, policeA的方向, policeB的速度, policeB的方向]"""
"""state=[policeA的X, policeA的Y, policeB的X, policeB的Y, thief的X, thief的Y]"""

__author__ = "Ziyuan Song"

import gymnasium as gym
from gymnasium import spaces
import numpy as np

OUTOFBAND_REWARD = -1000.0
Victory_REWARD = 1000.0
Failure_REWARD = -1000.0
MAX_STEP_CNT = 40
DEG2RAD = np.pi / 180

class ChasingEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'render_fps': 5
    }

    def __init__(self):
        
        self.moving_police_size = 1
        self.moving_thief_size = 0.5 

        self.space_upper_limit = np.array([15.0, 15.0, 15.0, 15.0, 15.0, 15.0])
        self.space_lower_limit = np.array([-15.0, -15.0, -15.0, -15.0, -15.0, -15.0])
        self.state_upper_limit = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.state_lower_limit = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        self.action_upper_limit = np.array([1.0, 1.0, 1.0, 1.0])
        self.action_lower_limit = np.array([-1.0, -1.0, -1.0, -1.0])

        self.action_space = spaces.Box(self.action_lower_limit, 
                                       self.action_upper_limit, 
                                       dtype=np.float32)
        
        self.observation_space = spaces.Box(self.state_lower_limit, 
                                            self.state_upper_limit, 
                                            dtype=np.float32)
        
        self.state = None
        self.moving_police_position_A = None
        self.moving_police_position_B = None
        self.screen = None
        self.clock = None
        self.render_mode = 'human'
        self.step_count = 0

    def step(self, action):

        self.step_count += 1
        truncated = False
        terminated = False

        # police_A
        norm_speed_A = np.clip(action[0], -1.0, 1.0) 
        speed_A = (norm_speed_A +1 ) * 1
        direction_A = (action[1] * 180.0 + 180.0) * DEG2RAD
        next_police_position_A = (self.moving_police_position_A 
                         + np.array([speed_A * np.cos(direction_A),
                                     speed_A * np.sin(direction_A)]))
        # police_B
        norm_speed_B = np.clip(action[2], -1.0, 1.0) 
        speed_B = (norm_speed_B +1) * 1
        direction_B = (action[3] * 180.0 + 180.0) * DEG2RAD
        next_police_position_B = (self.moving_police_position_B 
                         + np.array([speed_B * np.cos(direction_B),
                                     speed_B * np.sin(direction_B)]))
        
        #thief
        speed_thief = np.random.rand(1) * 1 
        speed_thief=speed_thief.item()
        direction = (np.random.rand(1) * 360) * DEG2RAD
        direction=direction.item()
        next_thief_position = (self.moving_thief_position  
                               + np.array([speed_thief * np.cos(direction),
                                     speed_thief * np.sin(direction)]))

        next_thief_position[0] = np.clip(next_thief_position[0], self.space_lower_limit[0]+4, -self.space_lower_limit[0]-4)
        next_thief_position[1] = np.clip(next_thief_position[1], self.space_lower_limit[1]+4, -self.space_lower_limit[1]-4)

        advantage_move_A = (self.get_distance(self.moving_thief_position, 
                                            self.moving_police_position_A) 
                          -self.get_distance(next_thief_position,
                                             next_police_position_A)  )

        advantage_move_B = (self.get_distance(self.moving_thief_position, 
                                            self.moving_police_position_B) 
                          -self.get_distance(next_thief_position,
                                             next_police_position_B)  )
        
        self.moving_thief_position = next_thief_position
        self.moving_police_position_A = next_police_position_A
        self.moving_police_position_B = next_police_position_B

        two_dist = min( self.get_distance(self.moving_thief_position,self.moving_police_position_A) ,
                            self.get_distance(self.moving_thief_position,self.moving_police_position_B) )
        
        if (two_dist <= 2):
            reward = Victory_REWARD
            terminated = True
        elif self.oob_detect(self.moving_police_position_A) or self.oob_detect(self.moving_police_position_B) :
            reward = OUTOFBAND_REWARD
            terminated = True
        elif self.step_count >= MAX_STEP_CNT:
            reward = Failure_REWARD
            truncated = True
            """
        elif advantage_move_A >= 0.2 and advantage_move_B >= 0.2 :
            reward = 50*(advantage_move_A + advantage_move_B )  
        else:
            reward = -10
            """
        else:
            reward = 100*(advantage_move_A + advantage_move_B )-100

        self.state = self._get_observation()
        info = {}

        return self.state, reward, terminated, truncated, info

    def oob_detect(self, position):
 
        oob_flag = (position[0] > self.space_upper_limit[0] 
                    or position[1] > self.space_upper_limit[1] 
                    or position[0] < self.space_lower_limit[0] 
                    or position[1] < self.space_lower_limit[1])
        
        return oob_flag

    def get_distance(self, coord1, coord2):

        dist = np.sqrt((coord1[0] - coord2[0]) ** 2 
                       + (coord1[1] - coord2[1]) ** 2)
        
        return dist

    def reset(self, seed=None, options={}):

        super().reset(seed=seed)

        if options is None:
            options = {}

        self.moving_police_position_A = np.array([0, 5])
        self.moving_police_position_B = np.array([0, -5])
        self.moving_thief_position = np.array([np.random.rand()*30-15, np.random.rand()*30-15])
        self.step_count = 0
        
        self.state = self._get_observation()

        info = {}
        return self.state, info
    
    def _get_observation(self):
        
        x1, y1 = self.moving_police_position_A
        x1_norm = x1 / self.space_upper_limit[0]
        y1_norm = y1 / self.space_upper_limit[0]

        x2, y2 = self.moving_police_position_B
        x2_norm = x2 / self.space_upper_limit[0]
        y2_norm = y2 / self.space_upper_limit[0]

        x3, y3 = self.moving_thief_position
        x3_norm = x3 / self.space_upper_limit[0]
        y3_norm = y3 / self.space_upper_limit[0]

        observation = np.array([x1_norm, y1_norm, x2_norm, y2_norm, x3_norm, y3_norm], dtype=np.float32)

        return observation

    def render(self, mode='human'):

        screen_width = 900
        screen_height =900

        world_width = 30.0  # Display from -4 to 4.
        scale = screen_width / world_width

        def cord_transform(position, scale):
            x = (position[0] + 15.0) * scale
            y = (position[1] + 15.0) * scale
            return [x, y]
        
        def draw_circle(x, y, radius, color):
            gfxdraw.aacircle(self.surf, x, y, radius, color)
            gfxdraw.filled_circle(self.surf, x, y, radius, color)
       
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            print("pygame is not installed")
            print("run `pip install pygame==2.5.2`")
            raise ImportError

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (screen_width, screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((screen_width, 
                                              screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Set background
        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))
        
        x, y = [int(i) for i in cord_transform(self.moving_thief_position,
                                               scale)]
        draw_circle(x, y, int(0.5 * scale), (0, 0, 0))

        x, y = [int(i) for i in cord_transform(self.moving_police_position_A,
                                               scale)]
        draw_circle(x, y, int(1 * scale), (129, 132, 203))

        x, y = [int(i) for i in cord_transform(self.moving_police_position_B,
                                               scale)]
        draw_circle(x, y, int(1 * scale), (129, 132, 203))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), 
                axes=(1, 0, 2)
            )

    def close(self):
        
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

        return super().close()


if __name__ == "__main__":

    env = ChasingEnv()
 
    obs = env.reset()
    episodic_reward = 0.0

    model = DDPG("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=30000)
    model.save("szy")

    del model # remove to demonstrate saving and loading
    model = DDPG.load("szy")

    for i in range (20):
        obs, info = env.reset()
        done = False
        episodic_reward = 0.0
        while not done:
            env.render("human")
            act, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(act)
            done = (terminated or truncated)
            episodic_reward += reward

        env.render("human")
        print("Reward this episode is %f" % episodic_reward)
        print("Steps this episode is %d" % env.step_count)





