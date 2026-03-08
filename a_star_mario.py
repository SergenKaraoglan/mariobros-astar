import numpy as np
import heapq
import time
from nes_py.wrappers import JoypadSpace
from nes_py._image_viewer import ImageViewer
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT

class Node:
    def __init__(self, g, h, info, parent, action, frames):
        self.g = g
        self.h = h
        self.f = g + h * 5.0 # strongly greedy
        self.info = info
        self.parent = parent
        self.action = action
        self.frames = frames

    def __lt__(self, other):
        return self.f < other.f

def get_nes_env(env):
    while hasattr(env, 'env'):
        env = env.env
    return env

def get_state_key(nes_env, info):
    x = info.get('x_pos', 0)
    y = info.get('y_pos', 0)
    vx = nes_env.ram[0x57]
    vy = nes_env.ram[0x9F]
    status = info.get('status', 'small')
    grid_x = x // 8
    grid_y = y // 8
    enemy_hash = hash(nes_env.ram[0x0F:0x20].tobytes())
    return (grid_x, grid_y, vx, status, enemy_hash)

def main():
    print("Initializing environment...")
    
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    nes_env = get_nes_env(env)
    
    viewer = ImageViewer(
        "Mario A* Agent Preview",
        256,
        240,
        monitor_keyboard=False
    )

    print("Resetting environment to start level...")
    obs, start_info = env.reset()
    start_time = time.time()
    
    # Initialize variables
    for _ in range(50):
        obs, _, _, _, start_info = env.step(0)
        viewer.show(obs)
        time.sleep(0.01)
    
    GOAL_X = 3100
    total_actions_taken = 0
    FRAMES_PER_ACTION = 8
    
    try:
        while True:
            # Anchor to this exact frame!
            nes_env._backup()
            
            if start_info['x_pos'] >= GOAL_X or start_info.get('life', 2) < 2:
                print("Goal reached or died!")
                break
                
            open_list = []
            start_node = Node(0, max(0, GOAL_X - start_info['x_pos']), start_info, None, None, 0)
            heapq.heappush(open_list, start_node)
            
            visited = set()
            visited.add(get_state_key(nes_env, start_info))
            
            nodes_expanded = 0
            best_node = start_node
            max_x = start_info['x_pos']
            search_start_time = time.time()
            
            # Ultra short search horizon!
            # 10 expansions drops max path depth to 4
            # 4 depth * 8 frames/depth * 10 nodes = 320 max frames to simulate!
            # Since step runs at 600fps, this finishes in 0.5 seconds inherently!
            while open_list and nodes_expanded < 10:
                current = heapq.heappop(open_list)
                nodes_expanded += 1
                
                if current.info['x_pos'] > max_x:
                    max_x = current.info['x_pos']
                    best_node = current
                    
                if current.info['x_pos'] >= GOAL_X:
                    best_node = current
                    break
                    
                # We only search moving RIGHT+B (3), RIGHT+A+B (4)
                for action in [3, 4]:
                    env.reset()
                    
                    path = []
                    curr = current
                    while curr.parent is not None:
                        path.append(curr.action)
                        curr = curr.parent
                    path.reverse()
                    
                    for past_action in path:
                        for _ in range(FRAMES_PER_ACTION):
                            env.step(past_action)
                            
                    dead = False
                    for _ in range(FRAMES_PER_ACTION):
                        _, _, done, truncate, info = env.step(action)
                        if done or info.get('life', 2) < current.info.get('life', 2):
                            dead = True
                            break
                            
                    if dead:
                        continue
                        
                    key = get_state_key(nes_env, info)
                    if key not in visited:
                        visited.add(key)
                        h = max(0, GOAL_X - info['x_pos'])
                        new_node = Node(current.g + FRAMES_PER_ACTION, h, info, current, action, FRAMES_PER_ACTION)
                        heapq.heappush(open_list, new_node)
            
            if best_node.parent is None:
                action_to_take = 1 # Force right if completely stuck
            else:
                curr = best_node
                while curr.parent.parent is not None:
                    curr = curr.parent
                action_to_take = curr.action
            
            elapsed_search = time.time() - search_start_time
            print(f"MPC Execution Step {total_actions_taken}: Searched {nodes_expanded} nodes in {elapsed_search:.2f}s. Selected action {action_to_take} predicting X={best_node.info['x_pos']}")
            
            # Apply the chosen action PERMANENTLY
            env.reset()
            for _ in range(FRAMES_PER_ACTION):
                obs, _, execution_done, execution_truncate, start_info = env.step(action_to_take)
                viewer.show(obs)
                time.sleep(0.016)
                if execution_done:
                    break
            
            total_actions_taken += 1
            
    except KeyboardInterrupt:
        print("Interrupted.")
        
    print(f"Finished. Reached X={start_info['x_pos']}")
    viewer.close()
    env.close()

if __name__ == '__main__':
    main()
