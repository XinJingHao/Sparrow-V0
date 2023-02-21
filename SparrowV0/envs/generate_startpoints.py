import pygame
import numpy as np
import os

def main(map_name):
    pygame.init()
    pygame.display.init()
    window_size = 366
    window = pygame.display.set_mode((window_size, window_size))
    clock = pygame.time.Clock()
    # canvas = pygame.Surface((window_size, window_size))
    map = pygame.image.load(os.getcwd() + '/train_maps/' + map_name + '.png')
    click_times = -1
    start_points = []
    while True:
        ev = pygame.event.get()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_ESCAPE]:
            start_points_np = np.array(start_points)
            np.save('train_maps_startpoints/'+map_name+'.npy',start_points_np)
            print('Start_points Saved.')
            print(start_points_np)
            break

        for event in ev:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                pos_wd = list(pos) #[x_wd,y_wd]
                pos_wd[1] = window_size-pos_wd[1] #grid2world

                click_times += 1
                if click_times % 2 == 0:
                    # 放入位置
                    start_points.append(pos_wd+[0])
                    last_pos = pos

                    # 小车碰撞阈值
                    pygame.draw.circle(
                        map,
                        (64,64,64),
                        pos,
                        20,
                    )

                    # 小车车体
                    pygame.draw.circle(
                        map,
                        (200, 128, 250),
                        pos,
                        9,
                    )
                else:
                    #画方向
                    pygame.draw.line(
                        map,
                        (0, 255, 255),
                        last_pos,
                        pos,
                        width=2
                    )
                    #算角度
                    dy = pos_wd[1] - start_points[-1][1]
                    dx = pos_wd[0] - start_points[-1][0]
                    if dy>0 and dx>0:
                        theta = np.arctan(dy/dx)
                    elif dy>0 and dx<0:
                        theta = np.pi + np.arctan(dy/dx)
                    elif dy<0 and dx<0:
                        theta = np.pi + np.arctan(dy / dx)
                    else: #dy<0 and dx>0
                        theta = 2*np.pi + np.arctan(dy / (dx+1e-6))

                    if theta > 2*np.pi : theta -= 2*np.pi
                    if theta < 0: theta += 2*np.pi

                    # 放入
                    start_points[-1][2] = theta
                    print(start_points[-1])


        window.blit(map, map.get_rect())
        pygame.event.pump()
        pygame.display.update()
        clock.tick(100)


if __name__ == '__main__':
    main(map_name='map0')