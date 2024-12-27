from torch import optim
import torch


class Trainer:
    def __init__(self, env, model, optimizer, replay_buffer, batch_size: int):
        self.gflownet = model
        self.optimizer = optimizer(model.parameters())
        self.replay_buffer = replay_buffer
        self.env = env

        self.visited_terminating_states = env.States.from_batch_shape((0,))
        self.states_visited = 0

        self.batch_size = batch_size

    def step(self):
        trajectories = self.gflownet.sample_trajectories(self.env, n_samples=self.batch_size)
        training_samples = self.gflownet.to_training_samples(trajectories)
        if self.replay_buffer is not None:
            with torch.no_grad():
                self.replay_buffer.add(training_samples)
                training_objects = self.replay_buffer.sample(
                    n_trajectories=self.batch_size
                )
        else:
            training_objects = training_samples

        self.optimizer.zero_grad()
        loss = self.gflownet.loss(self.env, training_objects)
        loss.backward()
        self.optimizer.step()

        self.visited_terminating_states.extend(trajectories.last_states)

        self.states_visited += len(trajectories)

        return loss.item(), self.states_visited
