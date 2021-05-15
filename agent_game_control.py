from game_env import GameEnv, show_image

class Control(object):
    def __init__(self, env):
        self.env = env
        self.jump() #we should start a game by jumping
        self.actions = [self.jump, self.crouch, self.do_nothing]

    def jump(self):
        self.env.jump()
    
    def crouch(self):
        self.env.crouch()

    def do_nothing(self):
        pass

    def choose_action(self, action_idx):
        self.actions[action_idx]()

    def is_crashed(self):
        return self.env.is_crashed()


class GameState(object):
    def __init__(self, agent, debug = True):
        self.agent = agent
        self.display = None
        if debug:
            self.display = show_image()
            self.display.__next__()
    
    def get_state(self, action_idx):
        sc = self.agent.env.get_score()
        score = 0 if sc == "" else int(sc)
        reward = 0.1
        is_gameover = False
        self.agent.choose_action(action_idx)
        screen = self.agent.env.capture_screen()
        if self.display is not None:
            self.display.send(screen)
        if self.agent.env.is_crashed():
            is_gameover = True
            reward = -1
            self.agent.env.restart_game()
        return screen, reward, is_gameover, score