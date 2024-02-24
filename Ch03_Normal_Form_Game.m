% Two agents
% Each agent has two actions: {a, b}

% Setup the game
Pd_Ag1 = [0.5 0.5]; % Probability distribution for actions of agent 1
Pd_Ag2 = [0.5 0.5];

% Randomize
rng("shuffle");

% Select action for agent 1
if rand <= Pd_Ag1(1)
    A1 = 1;
else
    A1 = 2;
end

% Select action for agent 2
if rand <= Pd_Ag2(1)
    A2 = 1;
else
    A2 = 2;
end

reward = Reward(A1, A2);

fprintf('(%d, %d, %d)\n', A1, A2, reward);

function reward = Reward(a1, a2)
    reward = a1 + a2;
end
