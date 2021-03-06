# D2 TPred : Discontinuous Dependency for Trajectory Prediction under Traffic Lights
# How was the data collected?
The data in VTP-TL comes from at urban intersections with traffic lights is used to predict vehicles trajectory in different times of day and provides a broad range of real-world driving scenarios. We use drones to hover at 70 to 120 meters above the traffic intersections, as statically as possible, to record vehicle trajectories passing through the area with a bird’s-eye view in the daytime of the non-rush hours, rush hours, and the evening.


<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/drone.png" width="780" height="312" alt=" "/><br/>
</div>

# Where was the data collected?
We choose 3 different traffic intersections, including crossroad, T-junction, and roundabout scenarios. In these scenario, they own the different number of roads and traffic lights, and cause to different movement behaviors for vehicles.

<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/scenarios.png" width="762" height="628" alt=" "/><br/>
</div>

# Summary of the Dataset 
In the [VTP-TL](https://pan.baidu.com/s/1gAdWP58RCKl0RrsvtQotpw) dataset, we have collected data from 3 different categories of traffic scenarios using drones. The summary of the data is listed in the following table. 

<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/summary.png" width="772" height="503" alt=" "/><br/>
</div>

# Included Materials
For the 3 recording scenarios, we include 2 files for each scenarios: 
1. The sample of video clips (xxx.mp4) 
2. Recorded vehicle trajectory file (xxx.txt) 
where, we provide trajectories information in pixel.

# Recorded Vehicle Trajectory files (xxx.txt)
**F_id:** column 1. For each agent (per Agent_id), frame_id represents the frames the agent appears in the video.    
**A_id:** column 2. For each xxx.txt file, the Agent_id starts from 0, and represent the ID of the agent.   
**x:** column 3, the x position of the agent at each frame. The unit is pixel.     
**y:** column 4, the y position of the agent at each frame. The unit is pixel.   
**Lane_id:** column 5, For each xxx.txt file, the Lane_id starts from 0, and represent the ID of the traffic lane.   
**pa:** column 6, For each xxx.txt file, the inperception is set as 0 or 1, and represent whether vehicle locates in the influencing area of traffic light.   
**f:** column 7, For each xxx.txt file, the isfirstobj is set as 0 or 1, and represent whether vehicle is the first agent in the influencing area of traffic light.   
**Lig_id:** column 8, For each xxx.txt file, the Lig_id starts from 0, and represent the ID of the traffic light.   
**ls:** column 9, For each xxx.txt file, the Ls is set as 0, 1 and 2, and represents the state of traffic light.   
**mb:** column 10, For each xxx.txt file, the Mb is set as 0, 1 and 2, and represents the movement behaviors of vehicle.   
**lt:** column 11, For each xxx.txt file, the Ldurtime represents the durtime of traffic light.   

**Example:**
<div align=center>
<img src="https://github.com/VTP-TL/D2-TPred/blob/main/smaple.png" alt=" "/><br/>
</div>

