<center> <h1> Squash Drive Analyser </h1> </center>

<p align="center"><img width=15% src="https://user-images.githubusercontent.com/75072722/181820844-8c8e3773-a489-4a7b-8ad0-133aea8a7103.png"></p>


## Basic Overview
Use videos of yourself hitting straight drives and automatically collect accuracy data from a session.
Save the data and track progress over time!

## Software demonstration

https://user-images.githubusercontent.com/75072722/181812643-88f6b03a-2d7f-4083-bea4-656a6f7fb875.mp4

See **[here](./resources/HOWTO.md)** for detailed how-to-use instructions.

## Installation
**Pre-requisite:** Download and install [Python3](https://www.python.org/downloads/)
1. Clone or download this repository
2. Install the required packages:
```bash
pip3 install -r requirements.txt
```

## Usage
```bash
python3 main.py
```

## Analysis accuracy
The software relies on the user to mark the service box and the rear boundary of the court and uses these as parameters for computing the ball bounce location. Therefore, it is advisable to place all markers as accurately as possible.


## Camera placement
* The video must be captured from behind the court
* The camera must be static during the video

