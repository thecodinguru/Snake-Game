# Snake-Game

<B> Hello there! </B>

I have created a snake game using PyGame.
In this gameof Snake, you can move in three directions: left, right or forward. This will be predicted by our neural network.

The neural network contains 9 input layers with two hidden layers, each with 24 nodes and a single output layer.

A Tanh function will used in this game as an activation function producing the prediction of: -1,0,1 [left,forward,right]

The 9 input layer will have values from these 9 questions:

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

The answers to these problems can also be solved with the help of reinforcement learning as well.

Snake Game Programming Method:
1. Randomly assign neural network weights and pass the input layer values
2. The neural network will predict a value and move in that direction 
3. Repeat steps 1 and 2 until snake gets a food or ignore (dies or in a loop)
4. Store the neural network weight information in a list
5. Try the above steps for random amount (N) numbers of snakes and select n smart snakes from the pool
6. Now take those n smart snakes and test there performance
7. Using the highest performer, it will create a threshold score which will help us to select more smart snakes 
8. Use the concept of Genetic Algorithm and mutate each snakes with self/each other and create more child snakes

NOTE: I randomly mutated parent snake weights with 1%,2%,3%,4%,5%,7% deviation and created 15 child snakes and also a child snake who is average of all parent snakes.

      JUST PLAY WITH THE MUTATIONS AS MUCH AS YOU WANT! HAVE FUN! MAKE AS MANY OF YOUR COMBINATIONS AS POSSIBLE!

9. Test the child snakes performance and again select the top performers and repeat from step 7
