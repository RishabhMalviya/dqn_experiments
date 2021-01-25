import imageio
import os


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    """Takes a list of frames (each frame can be generated with the `env.render()` function from OpenAI gym)
    and converts it into GIF, and saves it to the specified location.
    Code adapted from this gist: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

    Args:
        frames (list): A list of frames generated with the env.render() function
        path (str, optional): The folder in which to save the generated GIF. Defaults to './'.
        filename (str, optional): The target filename. Defaults to 'gym_animation.gif'.
    """
    imageio.mimwrite(os.path.join(path, filename), frames, fps=60)