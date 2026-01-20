'''
Remote Sensing Visualizer
Main application entry point

Last modified: Jan 2026
'''

from app.visualizer import Visualizer
import multiprocessing

def main():
    multiprocessing.freeze_support()

    app = Visualizer()
    app.mainloop()

if __name__ == '__main__':
    main()