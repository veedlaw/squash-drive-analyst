# Squash routine analysis software (NPRG045)
This project attempts to tackle the gap of existing squash analysis tools by using computer
vision techniques to automate the collection of shot data of a common squash training drill “rotating
drives” (see references). The main goal of this project is to provide relevant data and success
metrics based on the player’s shot location to the end user.

The software will allow the user to load a previously captured video via a graphical user
interface and run the shot analysis program on the video. Upon completion of the program, the user
will receive:

• The video with the ball clearly marked.

• A heat-map image of the ball bounce locations individualized per player.

• An image marking in which sections of the back court their shots landed percent-wise, individualized per player.



## Timetable

| Date               |Milestone                  |Method of presentation                |Status     |
|----------------|-------------------------------|--------------------------------------|------------
|~~01/05/2021~~  | ~~GUI file selection~~                         |~~Code in repository~~|✅        |                    
|01/06/2021      | Complete implementation of pre-processing stage| Code in repository  |✅         |
|01/07/2021      | Complete implementation of ball detection stage| Code in repository  |✅         |
|01/08/2021	     | Complete implementation of ball tracking stage | Code in repository  |✅         |
|01/09/2021	     | Complete implementation of bounce detection    | Code in repository  |           |
|14/09/2021	     | Implemented outputs of the application	      | Presentation        |           |
|20/09/2021		 | Final version						          | Demonstration       |           |


### References
• A more detailed description of the game of squash is provided here:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - https://en.wikipedia.org/wiki/Squash_(sport)

• OpenCV website:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - https://opencv.org,OpenCV team, 2021

• TKinter reference:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - https://en.wikipedia.org/wiki/Tkinter

• Video example of rotating drives squash routine:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - https://www.youtube.com/watch?v=5XTiiUEtuag
