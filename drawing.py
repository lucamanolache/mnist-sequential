import pygame
import pygame.freetype

import numpy as np

import torch

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 560, 560
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Drawing App")

# Set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the drawing variables
drawing = False
last_pos = (0, 0)

# Set up the font
font_size = 32
font = pygame.freetype.Font(None, font_size)

# Create a separate surface for drawings
drawing_surface = pygame.Surface((width, height))
drawing_surface.fill(BLACK)

# Game loop
running = True
diffs = [[]]
guess = None

# Load model
model = torch.jit.load("models/lstm-trained.pt")
model.eval()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear the screen when "C" key is pressed
                drawing_surface.fill(BLACK)
                pygame.display.flip()
                diffs = [[]]
                guess = None
            if event.key == pygame.K_q:
                running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
                last_pos = pygame.mouse.get_pos()

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
                diffs.append([])

        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = pygame.mouse.get_pos()
                if last_pos is not None:
                    pygame.draw.line(drawing_surface, WHITE, last_pos, current_pos, 5)

                if (
                    int(current_pos[0] / 20) - int(last_pos[0] / 20) != 0
                    or int(current_pos[1] / 20) - int(last_pos[1] / 20) != 0
                ):
                    diffs[-1].append(
                        [
                            int(current_pos[0] / 20) - int(last_pos[0] / 20),
                            int(current_pos[1] / 20) - int(last_pos[1] / 20),
                            0,
                            0,
                        ]
                    )
                    guess = ""
                    for x in diffs:
                        x = np.array(x)
                        x = torch.from_numpy(x).float().cuda()
                        guess += str(model(x).argmax().cpu().item())

                last_pos = current_pos

    # Update the display
    screen.fill(BLACK)
    screen.blit(drawing_surface, (0, 0))

    # Render and display the text
    if guess is not None:
        text_surface, _ = font.render("{}".format(guess), WHITE)
        screen.blit(text_surface, (10, 10))

    pygame.display.flip()

# Quit the game
pygame.quit()
