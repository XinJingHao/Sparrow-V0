import SparrowV0
import pygame
import gym

def main():
    env = gym.make('Sparrow-v0', render_mode='human', render_speed='fast')
    while True:
        env.reset()
        done = False
        while not done:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: a = 0
            elif keys[pygame.K_UP]: a = 2
            elif keys[pygame.K_RIGHT]: a = 4
            else: a = 5

            s, r, dw, tr, info = env.step(a)
            done = dw + tr



if __name__ == '__main__':
    main()