import base64
import io
import csv
import pygame

def visualize_csv(input_path, n):
    """
    Visualizes the first n states from a processed CSV file.
    """
    pygame.init()

    with open(input_path, 'r') as infile:
        reader = csv.DictReader(infile)
        for i, row in enumerate(reader):
            if i >= n:
                break

            print(f"Displaying state {i+1}/{n}")
            base64_str = row['state']
            img_data = base64.b64decode(base64_str)
            img_byte_arr = io.BytesIO(img_data)
            
            img_surface = pygame.image.load(img_byte_arr)
            
            screen = pygame.display.set_mode(img_surface.get_size())
            screen.blit(img_surface, (0, 0))
            pygame.display.flip()

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            running = False
            
    pygame.quit()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("-n", type=int, default=5, help="Number of states to visualize.")
    args = parser.parse_args()
    visualize_csv(args.input_csv, args.n)
