#import logging
# import math

import cv2
import mujoco
import numpy as np
# import numpy.typing as npt

# from revolve2.simulation.scene import Scene, SimulationState
# from revolve2.simulation.simulator import RecordSettings

# from ._control_interface_impl import ControlInterfaceImpl

#from _custom_mujoco_viewer import CustomMujocoViewer
# from ._scene_to_model import scene_to_model
# from ._simulation_state_impl import SimulationStateImpl

#import mujoco
import numpy as np
import os
import sys

# Settings
fps = 60
video_directory = os.getcwd()
pathpkl = "C:\\Users\\niels\\Downloads\\GRN_runs\\Rerun_6_2\\PKL"
pathxml = "C:\\Users\\niels\\Downloads\\GRN_runs\\Rerun_6_2\\XML"


# Get the directory of the current file
current_directory = os.path.dirname(__file__)

# Append the directory containing the module to the Python path
custom_module_directory = current_directory
sys.path.append(custom_module_directory)

# Now you can import the module
from _custom_mujoco_viewer import CustomMujocoViewer

import pickle
import time


def get_files_in_directory(directory):
    files = []
    # Iterate through all the files in the directory
    for file in os.listdir(directory):
        # Check if the path is a file (not a directory)
        if os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files

def get_model(path):
    with open(path, 'r') as file:
        xml_data = file.read()
    model = mujoco.MjModel.from_xml_string(xml_data)
    return model

def get_data(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def write_video(viewer, video):
    # Initialize image
    img = np.empty((viewer.viewport.height, viewer.viewport.width, 3),
                    dtype=np.uint8,)
    # Read pixels
    mujoco.mjr_readPixels(
        rgb=img,
        depth=None,
        viewport=viewer.viewport,
        con=viewer.ctx,
    )
    # Flip image
    img = np.flip(img, axis=0)  # img is upside down initially
    # Write image
    video.write(img)
    return video

def simulate_scene():    
    # ---- Get Files and Timepoints
    # Get file names
    xmls = get_files_in_directory(pathxml)
    pkls = get_files_in_directory(pathpkl)
    # Get timepoints
    timepoints_xml = []
    for xml in xmls:
        splitted_xml = xml.split("_")[1].split(".")
        if not((splitted_xml[0] == "0") and (splitted_xml[1] == "xml")):
            timepoints_xml.append(".".join(splitted_xml[0:2]))
        else:
            timepoints_xml.append(splitted_xml[0])
    timepoints_xml = np.array(timepoints_xml, dtype=float)

    timepoints_pkl = []
    for pkl in pkls:
        splitted_pkl = pkl.split("_")[1].split(".")
        if not((splitted_pkl[0] == "0") and (splitted_pkl[1] == "pkl")):
            timepoints_pkl.append(".".join(splitted_pkl[0:2]))
        else:
            timepoints_pkl.append(".".join(splitted_pkl[0]))
    timepoints_pkl = np.array(timepoints_pkl, dtype=float)
    
    # Get indices sorted by timepoints --> for minimum timepoint
    idx_xml = np.argsort(timepoints_xml)
    idx_pkl = np.argsort(timepoints_pkl)

    # ---- Set up viewer
    start_paused = False
    viewer = CustomMujocoViewer(
            get_model(pathxml + "\\" + xmls[idx_xml[0]]),
            get_data(pathpkl + "\\" + pkls[idx_pkl[0]]),
            start_paused=start_paused,
            render_every_frame=False,
        )
    
    # ---- Initialize video writer
    video_file_path = f"{video_directory}\\video.mp4"
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    video = cv2.VideoWriter(
            video_file_path,
            fourcc,
            fps,
            (viewer.viewport.width, viewer.viewport.height),
        )
    
    time.sleep(0.5)
    # ---- Render first frame
    viewer.render()
    video = write_video(viewer, video)
    tstart = time.time()
    print("Zoommmmm")
    while (time.time() - tstart) < 3:
        time.sleep(0.01)
        viewer.render()

    # ---- Simulate
    #t0 = time.time()
    #while True:
        #timepoint = time.time() - t0
        #print(timepoint)
        #idxt = np.argmin(abs(timepoints_xml - timepoint))
        #idxt2 = np.argmin(abs(timepoints_pkl - timepoint))
    for t in range(1, len(timepoints_xml)):
        # Get file names
        xml = xmls[idx_xml[t]]
        pkl = pkls[idx_pkl[t]]
        # Load files
        model = get_model(pathxml + "\\" + xml)
        data = get_data(pathpkl + "\\" + pkl)

        # Render
        viewer.model = model
        viewer.data = data
        viewer.render()
        video = write_video(viewer, video)

        #if timepoint >= timepoints_xml[idx_xml[-1]]:
        if t == (len(timepoints_xml) - 1):
            break

    
    #for t in range(1, len(timepoints_xml)):
        # # Get indices
        # idxXML = idx_xml[t]
        # idxPKL = idx_pkl[t]
    
        # # Get file names
        # xml = xmls[idxXML]
        # pkl = pkls[idxPKL]
        # # Load files
        # model = get_model(pathxml + "\\" + xml)
        # data = get_data(pathpkl + "\\" + pkl)
        
    # ---- Close viewer
    viewer.close()
    # ---- Release video
    video.release()
        



    # model, mapping = scene_to_model(
    #     scene, simulation_timestep, cast_shadows=cast_shadows, fast_sim=fast_sim
    # )
    # data = mujoco.MjData(model)

    # viewer = CustomMujocoViewer(
    #     model,
    #     data,
    #     start_paused=start_paused,
    #     render_every_frame=False,
    # )

    
    #viewer.render()

    #logging.info(f"Scene done.")


if __name__ == "__main__":
    simulate_scene()
