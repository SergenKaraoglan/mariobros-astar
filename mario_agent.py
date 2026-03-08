from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

def main():
    print("Initializing Super Mario Bros Environment...")
    # Initialize the environment with the first level of Super Mario Bros
    env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    print("Environment created successfully.")
    
    # Restrict the action space to simple movements (Right, Right+A, Left, etc.)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    print("JoypadSpace wrapped successfully.")

    done = True
    for step in range(200): # Run for 200 frames to test
        if done:
            state = env.reset()
        
        # Take a random action
        action = env.action_space.sample()
        state, reward, done, truncate, info = env.step(action)
        
        if step % 50 == 0:
            print(f"Frame {step}: Mario X Position = {info.get('x_pos', 'Unknown')}, Mario Y Position = {info.get('y_pos', 'Unknown')}")
            
    env.close()
    print("Test run completed successfully.")

if __name__ == '__main__':
    main()
