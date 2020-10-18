# Snake-Game

<B> Hello there! </B>

I have created the snake game with the help of PyGame.
A snake can move in three directions that is left, right or forward. This will be predicted by our neural network.
The neural network contains 9 input layers with two hidden layers each with 24 nodes and a single output layer.
Tanh function is used as an activation function as it can predict -1,0,1 [i.e. left,forward,right]

The nine input layer will have values from this 9 questions :
Is food on left?
Is food on right?
Is food straight?
Is wall on right?
Is wall on left?
Is wall forward?
Is Snake's body on left?
Is Snake's body on right?
Is Snake's body in forward direction?

There are many ways to solve this problem but I tried to solve it with simple neural network.
The problem can also be solved with the help of reinforcement learning.

Method:
1. Randomly assign neural network weights and pass the input layer values
2. The neural network will predict a value as per the values and move in that direction 
3. Repeat the steps 1 and 2 untill snake gets a food or ignore if it dies or in a loop
4. Store the neural network weight information in a list
5. Try the above steps for N numbers of snakes and select n smart snakes from the pool
6. Now take those n smart snakes test there performance
7. The highest performer will create a threshold score which will help us to select more smart snakes 
8. Use the concept of Genetic Algorithm and mutate each snakes with self/each other and create more child snakes
NOTE: I randomly mutated parent snake weights with 1%,2%,3%,4%,5%,7% deviation and created 15 child snakes and also a child snake who is average of all parent snakes. 
NOTE: JUST PLAY WITH MUTATIONS! HAVE FUN! MAKE YOUR OWN POSSIBLE COMBINATIONS
9. Test the child snakes performance and again select the top performers and repeat from step 7
