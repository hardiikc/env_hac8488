import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class HardikEnv(gym.Env):
    def __init__(self, grid_size=7, cell_size=100):
        """
        The Environment consists of Agent, Goal State, Hell State, Profit State, and a background image.
        The Agent is a businessman who is striving to establish himself in the commercial world.
        There are four Hell states; if an agent reaches them, he must restart from the beginning. 
        There is one Profit State, which is an additional incentive characterized as talents for business growth. 

        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.agent_state = np.array([0, 0])  # Agent Initial State
        self.goal_state = np.array([5, 3])  # Goal State
        self.profit_state = np.array([3, 3])  # Profit State 
        self.profit_collected = False
        self.hell_state = [np.array([4, 2]), np.array([4, 4]), np.array([5, 1]), np.array([5, 5])]  # Hell State Coordinates
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(4)
        
        # Set up the plot
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.ion()
        plt.show(block=False)

        # Load images
        self.agent_img = mpimg.imread('agent.png')  # Agent here is Businessman surviving in the competitive market
        self.goal_img = mpimg.imread('goal.png')  # Goal is to achieve victory in growing business
        self.profit_img = mpimg.imread('profit.png')  # Profit is the skills which a businessman gains to help grow the business
        self.hell_img = mpimg.imread('hell.png')  # Hell state are the competitors in the business, leading to restart with something new to succeed
        self.grid_img = mpimg.imread('grid.png')  # It's a simple background based on the recent trend of Technology
        
        self.action = None  # To store user action
    
    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.agent_state = np.array([0, 0])
        self.profit_collected = False
        return self.agent_state
    
    def step(self, action):
        """
        Take a step in the environment based on the given action.
        """
        if action == 0 and self.agent_state[1] < self.grid_size - 1:
            self.agent_state[1] += 1
        if action == 1 and self.agent_state[1] > 0:
            self.agent_state[1] -= 1
        if action == 2 and self.agent_state[0] < self.grid_size - 1:
            self.agent_state[0] += 1
        if action == 3 and self.agent_state[0] > 0:
            self.agent_state[0] -= 1
        
        # Check if the agent hits a hell state
        if any(np.array_equal(self.agent_state, hell_state) for hell_state in self.hell_state):
            reward = -1
            done = False
            self.agent_state = np.array([0, 0])  # Reset to start position
            info = "Hit the hell state, returned to start position"
            return self.agent_state, reward, done, info
        
        reward = 0
        done = np.array_equal(self.agent_state, self.goal_state)
        if done:
            reward = 10
            if self.profit_collected:
                reward += 5
        elif np.array_equal(self.agent_state, self.profit_state):
            reward = 5
            self.profit_collected = True
            done = False
            info = "Learned Skills to Grow"
        
        distance_to_goal = np.linalg.norm(self.agent_state - self.goal_state)
        info = f"Distance to goal is {distance_to_goal}"

        return self.agent_state, reward, done, info
    
    def render(self):
        """
        Render the environment.
        """
        self.ax.clear()

        # Calculate aspect ratio of the grid image
        grid_img_aspect_ratio = self.grid_img.shape[1] / self.grid_img.shape[0]

        # Calculate new extent based on the aspect ratio
        if grid_img_aspect_ratio > 1:  # Wider than tall
            extent = [-0.5, self.grid_size - 0.5, -0.5 / grid_img_aspect_ratio, (self.grid_size - 0.5) / grid_img_aspect_ratio]
        else:  # Taller than wide or square
            extent = [-0.5 * grid_img_aspect_ratio, (self.grid_size - 0.5) * grid_img_aspect_ratio, -0.5, self.grid_size - 0.5]

        # Plot the grid background
        self.ax.imshow(self.grid_img, extent=[-0.5, self.grid_size - 0.5, -0.5, self.grid_size - 0.5])
        
        # Plot the agent
        self.ax.imshow(self.agent_img, extent=[self.agent_state[0]-0.5, self.agent_state[0]+0.5,
                                               self.agent_state[1]-0.5, self.agent_state[1]+0.5])
        
        # Plot the goal
        self.ax.imshow(self.goal_img, extent=[self.goal_state[0]-0.5, self.goal_state[0]+0.5,
                                              self.goal_state[1]-0.5, self.goal_state[1]+0.5])
        
        # Plot the Profit or extra reward
        if not self.profit_collected:
            self.ax.imshow(self.profit_img, extent=[self.profit_state[0]-0.5, self.profit_state[0]+0.5,
                                                    self.profit_state[1]-0.5, self.profit_state[1]+0.5])
        
        # Plot the hell states
        for hell_state in self.hell_state:
            self.ax.imshow(self.hell_img, extent=[hell_state[0]-0.5, hell_state[0]+0.5,
                                                  hell_state[1]-0.5, hell_state[1]+0.5])
        
        # Add black border around the entire grid
        self.ax.add_patch(plt.Rectangle((-0.5, -0.5), self.grid_size, self.grid_size,
                                        fill=None, edgecolor='black', linewidth=2))
        
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")
        self.ax.axis('off')  # Turn off the axis
        plt.pause(0.1)

    def on_key(self, event):
        """
        Handle key press events for user input.
        """
        if event.key == 'up':
            self.action = 0
        elif event.key == 'down':
            self.action = 1
        elif event.key == 'right':
            self.action = 2
        elif event.key == 'left':
            self.action = 3

    def close(self):
        """
        Close the environment.
        """
        plt.close()

if __name__ == "__main__":
    env = HardikEnv()
    state = env.reset()
    done = False
    while not done:
        env.render()
        while env.action is None:
            plt.pause(0.1)  # Wait for the user to press a key
        action = env.action
        env.action = None
        agent_state, reward, done, info = env.step(action)
        print(f"State: {agent_state}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print("Congratulations on your business victory")
            break

    if env.profit_collected:
        print("Succeed in Business with Gaining Skills")
    else:
        print("Succeed in Business but need to Learn More")
    env.close()
