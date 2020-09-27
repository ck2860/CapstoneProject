import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
## Documentation TESTING for exploreFirst (LR = 0.005) policy file

## Documentation TESTING for the Contextual Bandit class
# this class would start with a contextual bandit, get a bandit,  pull an arm functions
class contextual_bandit():
    ## create a new Contextual Bandit
    # initialize with a state, bandits, number of bandits, and number of actions
    def __init__(self):
        # create a new Contextual Bandit
        self.state = 0 # The number of current bandit
        self.bandits = np.array(data)
        #         self.bandits = np.random.uniform(low=-10, high=10, size=(4,4))
        self.num_bandits = self.bandits.shape[0] # the bandit number  (3)
        self.num_actions = self.bandits.shape[1] # the action number  (4)

    ## Getting the bandit aka state
    # returns a random state for every time.
    def getBandit(self):
        self.state = np.random.randint(0,len(self.bandits))
        return self.state

    ## Pulling an arm
    # return a positive or negative reward
    def pullArm(self,action):
        #Get a random number.
        bandit = self.bandits[self.state, action]
        #             print(bandit)
        result = np.random.randn(1)
        #             print(result)
        if result > bandit:
            #return a positive reward.
            return 1
        else:
            #return a negative reward.
            return -1

## Documentation for the agent class
# this class sets up a contextual bandit agent via building a feed-forward in the network and training/updating the network.
class agent():
    ## create a new agent
    # The agent would have a learning rate, the size of a state, and the size of actions.
    def __init__(self, lr, s_size,a_size):
        self.state_in= tf.placeholder(tf.int32,shape=(1,))

        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        # layer
        state_in_OH = tf.one_hot(self.state_in, s_size)
        output = tf.layers.dense(state_in_OH, a_size, tf.nn.sigmoid, use_bias=False, kernel_initializer = tf.ones_initializer())
        self.output = tf.reshape(output,[-1])
        self.chosen_action = tf.argmax(self.output,0)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(tf.float32, shape=(1,))
        self.action_holder = tf.placeholder(tf.int32, shape=(1,))

        self.responsible_weight = tf.slice(self.output,self.action_holder,[1])
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

# Clear the Tensorflow graph.
tf.reset_default_graph()

# This is where we would load the bandits.
cBandit = contextual_bandit()
# Loading the agent, the learning rate is 0.005
myAgent = agent(lr=0.005,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)
# The weights for the network.
weights = tf.trainable_variables()[0]

# Set up the total amount of episodes to train the agent.
total_episodes = 10000
# Create a score board.
total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions])

## Setting the chance of exploration
#Set the chance of taking a random action.
e = 0.1

init = tf.global_variables_initializer()

fig, ax = plt.subplots()
# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        s = cBandit.getBandit() #Get a state from the environment.

        #Choose either a random action or one from our network. This is explore first policy
        # first 1,000 rounds, the agent would explore then exploit the rest.
        if i < (e*total_episodes):
            action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]}) # exploitative
        else:
            action = np.random.randint(cBandit.num_actions) #eploration

        reward = cBandit.pullArm(action) #Get our reward for taking an action given a bandit.

        #Update the network.
        feed_dict={myAgent.reward_holder:[reward],myAgent.action_holder:[action],myAgent.state_in:[s]}
        _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)

        #Update our running tally of scores.
        total_reward[s,action] += reward
        if i % 500  == 0:

            meanR = np.mean(total_reward,axis=1)
            B1 = plt.plot(i, meanR[0], 'r.')
            B2 = plt.plot(i, meanR[1], 'y.')
            B3 = plt.plot(i, meanR[2], 'g.')
            B4 = plt.plot(i, meanR[3], 'b.')
        i+=1

ax.plot([], [], marker=".",  color = "r", label='Bandit#1')
ax.plot([], [], marker=".", color = "y", label='Bandit#2')
ax.plot([], [], marker=".", color = "g", label='Bandit#3')
ax.plot([], [], marker=".", color = "b", label='Bandit#4')

right_flag = 0
wrong_flag = 0

for a in range(cBandit.num_bandits):
    print("The agent thinks action " + str(np.argmax(ww[a])+1) + " for bandit " + str(a+1) + " is the most efficient")
    if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
        print("...and it was right!")
        right_flag += 1
    else:
        print("...and it was wrong!")
        wrong_flag += 1

prediction_accuracy = (right_flag/(right_flag+wrong_flag))
print("Prediction Accuracy (%):", prediction_accuracy * 100)

ax.legend()
plt.title("Mean Reward via the Explore-First Strategy (LR = 0.005)")
plt.show()