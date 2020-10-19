import pygame
import sys
import random
import time

class Snake():
    def __init__(self):
        self.position = [100,50]
        self.body=[[100,50],[90,50],[80,50]]
        self.direction="RIGHT"
        self.changeDirectonTo = self.direction

    def changeDirTo(self,dir):
        if dir=="RIGHT" and not self.direction=="LEFT":
            self.direction="RIGHT"
        if dir=="LEFT" and not self.direction=="RIGHT":
            self.direction="LEFT"
        if dir=="UP" and not self.direction=="DOWN":
            self.direction="UP"
        if dir=="DOWN" and not self.direction=="UP":
            self.direction="DOWN"
    def move(self,foodPos):
        if self.direction=="RIGHT":
            self.position[0]+=10
        if self.direction=="LEFT":
            self.position[0]-=10
        if self.direction=="UP":
            self.position[1]-=10
        if self.direction=="DOWN":
            self.position[1]+=10
        self.body.insert(0,list(self.position))
        if self.position==foodPos:
            return 1
        else:
            self.body.pop()
            return 0
    def checkCollision(self):
        if self.position[0]>490 or self.position[0]<0:
            return 1
        elif self.position[1]>490 or self.position[1]<0:
            return 1
        for y in self.body[1:]:
            if self.position==y:
                return 1
        return 0
    def getHeadPos(self):
        return self.position
    def getBody(self):
        return self.body
    def getdirection(self):
        return self.direction


class FoodSpawer():
    def __init__(self):
        self.position=[random.randrange(1,50)*10,random.randrange(1,50)*10]
        self.isFoodOnScreen = True
    def spawnFood(self):
        if self.isFoodOnScreen==False:
            self.position=[random.randrange(1,50)*10,random.randrange(1,50)*10]
            self.isFoodOnScreen=True
        return self.position
    def setFoodOnScreen(self):
        self.isFoodOnScreen=False

import math
import numpy as np

def gameOver():
    pygame.quit()
    sys.exit()
temp_value=[]



class dirintial():
    def initial(self):
        i = random.randrange(-1,1)

        return i




class wallcheck():
    def rightwallcheck(self,spos,body):
        if (spos[0]+10)>490:
            return 1
        return 0
    def rightbodycheck(self,spos,body):
            for y in body[1:]:
                if spos[0]+10==y[0]:
                    return 1
            return 0


    def leftwallcheck(self,spos,body):
        if (spos[0]-10)<0:
            return 1
        return 0
    def leftbodycheck(self,spos,body):
        for y in body[1:]:
            if spos[0]-10==y[0]:
                return 1
        return 0


    def upwallcheck(self,spos,body):
        if (spos[1]-10)<0:
            return 1
        return 0
    def upbodycheck(self,spos,body):
            for y in body[1:]:
                if spos[1]-10==y[1]:
                    return 1
            return 0



    def downwallcheck(self,spos,body):
        if (spos[1]+10)>490:
            return 1
        return 0
    def downbodycheck(self,spos,body):
        for y in body[1:]:
            if spos[1]+10==y[1]:
                return 1
        return 0


min = 0
max=500
rf=0
lf=0
sf=0


class instance():


    def step(self,action):
         #food instance
        t=0
        wcheck=wallcheck()
        reward=0
        lw,fw,rw,lf,sf,rf=0,0,0,0,0,0
        while True:
            i= action
            spos = snake.getHeadPos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    gameOver()
            if i == -1 and snake.getdirection()=="RIGHT":
                snake.changeDirTo('UP')
            elif i == 1 and snake.getdirection()=="RIGHT":
                snake.changeDirTo('DOWN')
            elif i == -1 and snake.getdirection()=="LEFT":
                snake.changeDirTo('DOWN')
            elif i == 1 and snake.getdirection()=="LEFT":
                snake.changeDirTo('UP')
            elif i == -1 and snake.getdirection()=="DOWN":
                snake.changeDirTo('RIGHT')
            elif i == 1 and snake.getdirection()=="DOWN":
                snake.changeDirTo('LEFT')
            elif i == -1 and snake.getdirection()=="UP":
                snake.changeDirTo('LEFT')
            elif i == 1 and snake.getdirection()=="UP":
                snake.changeDirTo('RIGHT')
            elif i==0:
                snake.changeDirTo(snake.getdirection())
            body1 = snake.getBody()
            #IS IT STILL CLEAR AHEAD POINT / IS IT CLEAR IN RIGHT/ IS IT CLEAR IN LEFT
            if snake.getdirection()=="RIGHT":
                rw = wcheck.downwallcheck(spos,body1)
                lw = wcheck.upwallcheck(spos,body1)
                fw = wcheck.rightwallcheck(spos,body1)

                rbw = wcheck.downbodycheck(spos,body1)
                lbw = wcheck.upbodycheck(spos,body1)
                fbw = wcheck.rightbodycheck(spos,body1)

            elif snake.getdirection()=="LEFT":
                rw = wcheck.upwallcheck(spos,body1)
                lw = wcheck.downwallcheck(spos,body1)
                fw = wcheck.leftwallcheck(spos,body1)

                rbw = wcheck.upbodycheck(spos,body1)
                lbw = wcheck.downbodycheck(spos,body1)
                fbw = wcheck.leftbodycheck(spos,body1)

            elif snake.getdirection()=="UP":
                rw = wcheck.rightwallcheck(spos,body1)
                lw = wcheck.leftwallcheck(spos,body1)
                fw = wcheck.upwallcheck(spos,body1)

                rbw = wcheck.rightbodycheck(spos,body1)
                lbw = wcheck.leftbodycheck(spos,body1)
                fbw = wcheck.upbodycheck(spos,body1)

            elif snake.getdirection()=="DOWN":
                rw = wcheck.leftwallcheck(spos,body1)
                lw = wcheck.rightwallcheck(spos,body1)
                fw = wcheck.downwallcheck(spos,body1)

                rbw = wcheck.leftbodycheck(spos,body1)
                lbw = wcheck.rightbodycheck(spos,body1)
                fbw = wcheck.downbodycheck(spos,body1)

            foodPos = foodspawer.spawnFood()
            #IS FOOD STRAIGHT AHEAD / IS FOOD ON LEFT / IS FOOD ON RIGHT
            if snake.getdirection()=="RIGHT":
                if spos[1]==foodPos[1]:     #THIS MEANS FOOD IS IN SAME Y LINE
                    if foodPos[0]>spos[0] and foodPos[0]<max:
                        sf=1
                        rf=0
                        lf=0
                        print("FOOD IS STRAIGHT")
                    else:
                        sf=0 #THIS MEANS FOOD IS IN OPPOSITE DIRECTION
                        lf=0
                        rf=0
                if foodPos[1]>spos[1]: #RIGHT DIRECTION's RIGHT is DOWN DIRECTION
                    sf=0
                    rf=1
                    lf=0
                    print("FOOD ON RIGHT")
                if foodPos[1]<spos[1]: #RIGHT DIRECTION's LEFT is UP DIRECTION
                    sf=0
                    rf=0
                    lf=1
                    print("FOOD ON THE LEFT")
            elif snake.getdirection()=="LEFT":
                if spos[1]==foodPos[1]:     #THIS MEANS FOOD IS IN SAME Y LINE
                    if foodPos[0]<spos[0] and foodPos[0]>min:
                        sf=1
                        rf=0
                        lf=0
                        print("FOOD IS STRAIGHT")
                    else:
                        sf=0 #THIS MEANS FOOD IS IN OPPOSITE DIRECTION
                if foodPos[1]>spos[1]: #LEFT DIRECTION's LEFT is DOWN DIRECTION
                    sf=0
                    rf=0
                    lf=1
                    print("FOOD ON left")
                if foodPos[1]<spos[1]: #LEFT DIRECTION's RIGHT is UP DIRECTION
                    sf=0
                    rf=1
                    lf=0
                    print("FOOD ON THE right")
            elif snake.getdirection()=="UP":
                if spos[0]==foodPos[0]:     #THIS MEANS FOOD IS IN SAME X LINE
                    if foodPos[1]<spos[1] and foodPos[1]>min:
                        sf=1
                        rf=0
                        lf=0
                        print("FOOD IS STRAIGHT")
                    else:
                        sf=0 #THIS MEANS FOOD IS IN OPPOSITE DIRECTION
                if foodPos[0]<spos[0]: #UP DIRECTION's LEFT is LEFT DIRECTION
                    sf=0
                    rf=0
                    lf=1
                    print("FOOD ON left")
                if foodPos[0]>spos[0]: #LEFT DIRECTION's RIGHT is UP DIRECTION
                    sf=0
                    rf=1
                    lf=0
                    print("FOOD ON THE right")
            elif snake.getdirection()=="DOWN":
                if spos[0]==foodPos[0]:     #THIS MEANS FOOD IS IN SAME X LINE
                    if foodPos[1]>spos[1] and foodPos[1]<max:
                        sf=1
                        rf=0
                        lf=0
                        print("FOOD IS STRAIGHT")
                    else:
                        sf=0 #THIS MEANS FOOD IS IN OPPOSITE DIRECTION
                if foodPos[0]>spos[0]: #down DIRECTION's LEFT is RIGHT DIRECTION
                    sf=0
                    rf=0
                    lf=1
                    print("FOOD ON left")
                if foodPos[0]<spos[0]: #LEFT DIRECTION's RIGHT is UP DIRECTION
                    sf=0
                    rf=1
                    lf=0
                    print("FOOD ON THE right")

            if(snake.move(foodPos)==1):
                reward=1                                  #if got food
                t=1
                foodspawer.setFoodOnScreen()

            window.fill(pygame.Color(225,225,225))
            for pos in snake.getBody():
                pygame.draw.rect(window,pygame.Color(0,225,0),pygame.Rect(pos[0],pos[1],10,10))
            pygame.draw.rect(window,pygame.Color(225,0,0),pygame.Rect(foodPos[0],foodPos[1],10,10))



            if(snake.checkCollision()==1):
                reward=-1                                  #if Collision Happens
                dist=1
                done=True
                return lw,fw,rw,lf,sf,rf,reward,dist,done,t,lbw,fbw,rbw



            pygame.display.flip()
            fps.tick(60)




            x1,y1= snake.getHeadPos()
            x2,y2= foodPos
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) #find distance between food and headposition

            done=False

            return lw,fw,rw,lf,sf,rf,reward,dist,done,t,lbw,fbw,rbw






#CODING PART 2

class NeuralNet():
    def __init__(self):
        self.inputsize=9
        self.hiddensize1=24
        self.hiddensize2=24
        self.outputsize=1

        self.W1 = np.random.randn(self.inputsize,self.hiddensize1) # 6x12 weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddensize1,self.hiddensize2) # 6x12 weight matrix from input to hidden layer

        self.W3 = np.random.randn(self.hiddensize2,self.outputsize) #12x1 weight matrix from hidden layer to ouput layer

    def forward(self, X):
        #FORWARD PROPAGATION CODE
        self.z = np.dot(X,self.W1) #dot product of input X and first set of 12x24 matrix
        self.z2 = self.tanh(self.z)
        self.z3 = np.dot(self.z2,self.W2)  #hiddenlayer1
        self.z4 = self.tanh(self.z3)
        self.z5 = np.dot(self.z4,self.W3)
        o = self.tanh(self.z5)
        return o


    def sigmoid(self,s):
        #activation function
        return 1/(1+np.exp(-s))
    def tanh(self,x):
        return np.tanh(x)

    def sigmoidPrime(self,s):
        #derivative
        return s*(1-s)

    def tanh_deriv(self,x):
        return 1.0 - np.tanh(x)**2


    def train(self,X,y):
        o = self.forward(X)

        self.backward(X,y,o)

    def saveweights(self):
        return self.W1,self.W2,self.W3

    def initializeweight(self,w1,w2,w3):
        self.W1=w1
        self.W2=w2
        self.W3=w3

    def predict(self,X):
        print("Predicted data based on trained weights : ")
        p=self.forward(X)
        if p>0.5:
            return 1
        elif p<-0.5:
            return -1
        else:
            return 0
    def initializeweight2(self,w1,w2,w3,i):
        self.W1=w1*i
        self.W2=w2*i
        self.W3=W3*i

import pandas as pd
q=[]
p=[]

for generations in range(100):
    u = generations
    r=0
    owt1 = np.zeros((9,24))
    owt2 = np.zeros((24,24))
    owt3 = np.zeros((24,1))
    orientation=1
    change=1
    for i in range(1000):
        NN=NeuralNet()
        window = pygame.display.set_mode((500,500))
        pygame.display.set_caption("Wow Snake")
        fps = pygame.time.Clock()
        snake = Snake()   #snake instance
        state = instance()
        foodspawer = FoodSpawer()
        X =np.array([0,1,0,0,1,0,0,0,0])
        score = 0 # NEURAL NET SCORE
        temp=0 #variable to check loop
        loop=0 #variable to check loop
        if len(q)==50:
            break
        if r >0 :
            q.append([wt1,wt2,wt3])
        r=0
        k=0
        temp = 0
        temp2 = 0
        score=0
        while k<100:
            print("GENERATION : "+str(u)+ " NEURAL NET : "+str(i)+" STEP : "+str(k) + " SCORE : "+ str(score))
            print("Predicted output: "+str(NN.forward(X)))
            predictedoutput = NN.predict(X)
            lw,fw,rw,lf,sf,rf,reward,dist,done,t,lbw,fbw,rbw =state.step(predictedoutput)
            X=np.array([lw,fw,rw,lf,sf,rf,lbw,fbw,rbw])
            score+=t
            pygame.display.set_caption("WOW SNAKE | SCORE :" + str(score) + "Selected:" + str(len(q)))
            if reward==-1:
                k=2001
            if t==1:
                r+=1
                wt1,wt2,wt3 = NN.saveweights()
            k+=1

#END OF SELECTION

temporary = q
scoremonitor = []
scoremonitortemp = 0

for generations in range(100):
    u = generations
    r=0
    op=[]

    for i in range(len(q)):

        NN=NeuralNet()
        window = pygame.display.set_mode((500,500))
        pygame.display.set_caption("Wow Snake")
        fps = pygame.time.Clock()
        snake = Snake()   #snake instance
        state = instance()
        foodspawer = FoodSpawer()
        X =np.array([0,1,0,0,1,0,0,0,0])
        temp=0 #variable to check loop
        loop=0 #variable to check loop
        a = q[i]
        NN.initializeweight(a[0],a[1],a[2])
        r=0
        k=0
        temp = 0
        temp2 = 0
        score=0

        while k<5000:
            print("GENERATION : "+str(u)+ " NEURAL NET : "+str(i)+" STEP : "+str(k) + " SCORE : "+ str(score))
            print("Predicted output: "+str(NN.forward(X)))
            predictedoutput = NN.predict(X)
            lw,fw,rw,lf,sf,rf,reward,dist,done,t,lbw,fbw,rbw =state.step(predictedoutput)
            X=np.array([lw,fw,rw,lf,sf,rf,lbw,fbw,rbw])
            score+=t
            pygame.display.set_caption("WOW SNAKE | SCORE :" + str(score) + "Selected:" + str(len(q)))
            if reward==-1:
                k=5001
            if t==1:
                r+=1
                wt1,wt2,wt3 = NN.saveweights()
                temp=0
            if t==0 :
                temp+=1
            if temp>500:
                k=5001
            k+=1
        op.append([score,[wt1,wt2,wt3]])
        if score>scoremonitortemp:
            scoremonitortemp=score
            scoremonitor.append(score)
    op1=[]
    for i in range(len(op)):
        temp=op[i]
        a= temp[0]
        b=temp[1]
        w1,w2,w3 = b[0],b[1],b[2]
        if a>scoremonitortemp-5:
            op1.append([w1,w2,w3])
            op1.append([w1*1.01,w2*1.01,w3*1.01])
            op1.append([w1*0.99,w2*0.99,w3*0.99])
            op1.append([w1*1.02,w2*1.02,w3*1.02])
            op1.append([w1*0.98,w2*0.98,w3*0.98])

            op1.append([w1*1.01,w2,w3])
            op1.append([w1,w2*1.01,w3])
            op1.append([w1,w2,w3*1.01])
            op1.append([w1*1.07,w2*0.95,w3*1.01])
            op1.append([w1*1.03,w2*0.97,w3*1.03])

            op1.append([w1*0.98,w2*1.01,w3*0.98])
            op1.append([w1,w2*1.02,w3*0.99])
            op1.append([w1*0.97,w2*0.97,w3*0.97])
            op1.append([w1*1.03,w2*1.03,w3*1.03])
            op1.append([w1*1.07,w2*0.94,w3*1.06])


    if len(op)==1:
        scoremonitortemp=7
        temp = op[0]
        a= temp[0]
        b=temp[1]
        w1,w2,w3 = b[0],b[1],b[2]

        for i in range(1,len(op)):
            temp=op[i]
            a= temp[0]
            b=temp[1]
            w1 += b[0]
            w2 += b[1]
            w3 += b[2]
        op1.append([(w1/len(op)),(w2/len(op)),(w3/len(op))])
    elif len(op)==0:
        print("Finished")
        break
    else:
        temp = op[0]
        a= temp[0]
        b=temp[1]
        w1,w2,w3 = b[0],b[1],b[2]

        for i in range(1,len(op)):
            temp=op[i]
            a= temp[0]
            b=temp[1]
            w1 += b[0]
            w2 += b[1]
            w3 += b[2]
        op1.append([(w1/len(op)),(w2/len(op)),(w3/len(op))])
    q=op1
print(scoremonitor)
