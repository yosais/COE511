% Train DQN Agent to Balance Cart-Pole System
% https://www.mathworks.com/help/reinforcement-learning/ug/train-dqn-agent-to-balance-cart-pole-system.html

% Create the envrionment

env = rlPredefinedEnv("CartPole-Discrete");

% Get the observation specification information
% Obs = (Position of Cart, Velocity of Cart, Pole Angle, Pole Angle Derivative)
%     = (x, dx, theta, dtheta)

obsInfo = getObservationInfo(env);

% Get the action specification information
% Discrete action space of two possible force values to the cart: –10 or 10 N

actInfo = getActionInfo(env);

% Fix the random generator seed for reproducibility.
% When you run the program again, the same experiences are generated.

rng(0);

% Create the DQN agent
% The Q-function has observations as inputs and state-action values as outputs
% To approximate the Q-function, use a neural network with one input channel 
% (the 4-dimensional observed state vector) and one output channel with 
% two elements (one for the 10 N action, another for the –10 N action).
%
% See the diagram (architecture) of the neural network on slide #37

net = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(length(actInfo.Elements))
    ];

% Create the deep nueral network

net = dlnetwork(net);

% Create the value network, which can also be called the critic
% >>> Note: DQN has no actor. That is, the target network is not an actor
% >>> network

critic = rlVectorQValueFunction(net,obsInfo,actInfo);

% Check the critic with a random observation input

disp( getValue(critic,{rand(obsInfo.Dimension)}));

% Create the DQN agent

agent = rlDQNAgent(critic);

% Check the agent with a random observation input

disp( getAction(agent,{rand(obsInfo.Dimension)}) );

% Specify the DQN agent options, including training options for the critic

agent.AgentOptions.UseDoubleDQN = false;
agent.AgentOptions.TargetSmoothFactor = 1;
agent.AgentOptions.TargetUpdateFrequency = 4;
agent.AgentOptions.ExperienceBufferLength = 1e5;
agent.AgentOptions.MiniBatchSize = 256;
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-3;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

% Specify the training options
% Run one training session containing at most 1000 episodes, with each episode lasting at most 500 time steps.
% Stop training when the agent receives an moving average cumulative reward greater than 480. At this point, 
% the agent can balance the cart-pole system in the upright position.

trainOpts = rlTrainingOptions(...
    MaxEpisodes=1000, ...
    MaxStepsPerEpisode=500, ...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=480); 

doTraining = false;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load the pretrained agent for the example.
    load("MATLABCartpoleDQNMulti.mat","agent")
end

% Simulate the DQN agent
% If total reward resulting from simulation is greater than 480, then the
% DQN agent was able to balance the cart-pole

simOptions = rlSimulationOptions(MaxSteps=500);
experience = sim(env,agent,simOptions);

totalReward = sum(experience.Reward)



