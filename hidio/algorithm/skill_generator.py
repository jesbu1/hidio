# Copyright (c) 2020 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gin
import torch
import copy
import numpy as np
from hidio.utils import flatten

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.sac_algorithm import SacAlgorithm, SacLossInfo
from alf.algorithms.config import TrainerConfig
from alf.data_structures import TimeStep, Experience, namedtuple, AlgStep
from alf.data_structures import make_experience, LossInfo, StepType
from alf.networks import EncodingNetwork
import alf.nest.utils as nest_utils
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.utils.conditional_ops import conditional_update
from alf.utils import dist_utils, math_ops, common, losses, tensor_utils
from alf.nest.utils import NestConcat
from alf.networks.preprocessors import EmbeddingPreprocessor

SubTrajectory = namedtuple(
    'SubTrajectory', ["observation", "prev_action"], default_value=())

DiscriminatorTimeStep = namedtuple(
    'DiscTimeStep', [
        "step_type", "observation", "state", "env_id", "batch_info",
        "prev_action", "reward",
    ],
    default_value=())

DiscriminatorState = namedtuple(
    "DiscriminatorState",
    ["untrans_observation", "subtrajectory", "first_observation"],
    default_value=())

SkillGeneratorState = namedtuple(
    "SkillGeneratorState",
    ["discriminator", "steps", "rl_reward", "rl_discount", "rl", "skill"],
    default_value=())

SkillGeneratorInfo = namedtuple(
    "SkillGeneratorInfo", ["reward", "skill", "switch_skill", "steps"],
    default_value=())

def is_action_skill(skill_type):
    return "action" in skill_type


def get_discriminator_spec(skill_type, observation_spec, action_spec):
    if is_action_skill(skill_type):
        return action_spec
    else:
        return observation_spec


def get_subtrajectory_spec(num_steps_per_skill, observation_spec, action_spec):
    observation_traj_spec = TensorSpec(
        shape=(num_steps_per_skill, ) + observation_spec.shape)
    action_traj_spec = TensorSpec(
        shape=(num_steps_per_skill, ) + action_spec.shape)
    return SubTrajectory(
        observation=observation_traj_spec, prev_action=action_traj_spec)


@gin.configurable
class Discriminator(Algorithm):
    def __init__(self,
                 observation_spec,
                 action_spec,
                 skill_spec,
                 config: TrainerConfig,
                 skill_discriminator_ctor=EncodingNetwork,
                 skill_encoder_ctor=None,
                 observation_transformer=math_ops.identity,
                 optimizer=None,
                 sparse_reward=False,
                 debug_summaries=False,
                 num_steps_per_skill=3,
                 skill_type="state_difference",
                 name="Discriminator"):
        """If ``sparse_reward=True``, then the discriminator will only predict
        at the skill switching steps.
        """
        if skill_spec.is_discrete:
            assert isinstance(skill_spec, BoundedTensorSpec)
            skill_dim = skill_spec.maximum - skill_spec.minimum + 1
        else:
            assert len(
                skill_spec.shape) == 1, "Only 1D skill vector is supported"
            skill_dim = skill_spec.shape[0]

        supported_skill_types = [
            "state_concatenation",
            "state_difference",
            "state",
            "action",
            "action_difference",
            "state_action",
            "action_concatenation",
        ]
        assert skill_type in supported_skill_types, (
            "Skill type must be in: %s" % supported_skill_types)

        self._skill_type = skill_type

        subtrajectory_spec = get_subtrajectory_spec(
            num_steps_per_skill, observation_spec, action_spec)

        if skill_type == "state_concatenation":
            discriminator_spec = flatten(subtrajectory_spec.observation)
        elif skill_type == "action_concatenation":
            discriminator_spec = flatten(subtrajectory_spec.prev_action)
        else:
            discriminator_spec = get_discriminator_spec(
                skill_type, observation_spec, action_spec)

        input_preprocessors, preprocessing_combiner = None, None
        if is_action_skill(skill_type):
            # first project
            input_preprocessors = (None, None)
            preprocessing_combiner = NestConcat()
            discriminator_spec = (observation_spec, discriminator_spec)

        skill_encoder = None
        if skill_encoder_ctor is not None:
            step_spec = BoundedTensorSpec((),
                                          maximum=num_steps_per_skill,
                                          dtype='int64')
            skill_encoder = skill_encoder_ctor(
                input_preprocessors=(None,
                                     EmbeddingPreprocessor(
                                         input_tensor_spec=step_spec,
                                         embedding_dim=skill_dim)),
                preprocessing_combiner=NestConcat(),
                input_tensor_spec=(skill_spec, step_spec))
            if input_preprocessors is None:
                input_preprocessors = (None, )
                discriminator_spec = (discriminator_spec, )
            input_preprocessors = input_preprocessors + (EmbeddingPreprocessor(
                input_tensor_spec=step_spec, embedding_dim=skill_dim), )
            discriminator_spec = discriminator_spec + (step_spec, )
            skill_dim = skill_encoder.output_spec.shape[0]

        skill_disc_inputs = dict(
            input_preprocessors=input_preprocessors,
            preprocessing_combiner=preprocessing_combiner,
            input_tensor_spec=discriminator_spec)

        if skill_discriminator_ctor.__name__ == "EncodingNetwork":
            skill_disc_inputs["last_layer_size"] = skill_dim
        else:  # ActorDistributionNetwork
            skill_disc_inputs["action_spec"] = skill_spec

        skill_discriminator = skill_discriminator_ctor(**skill_disc_inputs)

        train_state_spec = DiscriminatorState(
            first_observation=observation_spec,
            untrans_observation=
            observation_spec,  # prev untransformed observation diff for pred
            subtrajectory=subtrajectory_spec)

        super().__init__(
            train_state_spec=train_state_spec,
            predict_state_spec=DiscriminatorState(
                subtrajectory=subtrajectory_spec,
                first_observation=observation_spec),
            config=config,
            optimizer=optimizer,
            debug_summaries=debug_summaries,
            name=name)

        self._skill_discriminator = skill_discriminator
        self._skill_encoder = skill_encoder
        # exp observation won't be automatically transformed when it's sampled
        # from the replay buffer. We will do this manually.
        self._observation_transformer = observation_transformer
        self._num_steps_per_skill = num_steps_per_skill
        self._sparse_reward = sparse_reward
        self._skill_dim = skill_dim
        self._high_rl = None

    def is_action_skill(self):
        return is_action_skill(self._skill_type)

    def set_high_rl(self, rl):
        self._high_rl = rl
        self._trainable_attributes_to_ignore = lambda: ["_high_rl"]

    def _predict_skill_loss(self, observation, prev_action, prev_skill, steps,
                            state):
        # steps -> {1,2,3}
        if self._skill_type == "action":
            subtrajectory = (state.first_observation, prev_action)
        elif self._skill_type == "action_difference":
            action_difference = prev_action - state.subtrajectory[:, 1, :]
            subtrajectory = (state.first_observation, action_difference)
        elif self._skill_type == "state_action":
            subtrajectory = (observation, prev_action)
        elif self._skill_type == "state":
            subtrajectory = observation
        elif self._skill_type == "state_difference":
            subtrajectory = observation - state.untrans_observation
        elif "concatenation" in self._skill_type:
            subtrajectory = alf.nest.map_structure(
                lambda traj: traj.reshape(observation.shape[0], -1),
                state.subtrajectory)
            if is_action_skill(self._skill_type):
                subtrajectory = (state.first_observation,
                                 subtrajectory.prev_action)
            else:
                subtrajectory = subtrajectory.observation

        if self._skill_encoder is not None:
            steps = self._num_steps_per_skill - steps
            if not isinstance(subtrajectory, tuple):
                subtrajectory = (subtrajectory, )
            subtrajectory = subtrajectory + (steps, )
            with torch.no_grad():
                prev_skill, _ = self._skill_encoder((prev_skill, steps))

        if isinstance(self._skill_discriminator, EncodingNetwork):
            pred_skill, _ = self._skill_discriminator(subtrajectory)
            loss = torch.sum(
                losses.element_wise_squared_loss(prev_skill, pred_skill),
                dim=-1)
        else:
            pred_skill_dist, _ = self._skill_discriminator(subtrajectory)
            loss = -dist_utils.compute_log_probability(pred_skill_dist,
                                                       prev_skill)
        return loss

    def _update_state_if_necessary(self, switch_skill, new_state, state):
        return alf.nest.map_structure(
            lambda ns, s: torch.where(
                torch.unsqueeze(switch_skill, dim=-1), ns, s), new_state,
            state)

    def _clear_subtrajectory_if_necessary(self, subtrajectory, switch_skill):
        def _clear(subtrajectory):
            zeros = torch.zeros_like(subtrajectory)
            return torch.where(
                switch_skill.reshape(-1, 1, 1).expand(-1, *zeros.shape[1:]),
                zeros, subtrajectory)

        return alf.nest.map_structure(_clear, subtrajectory)

    def update_subtrajectory(self, time_step, state):
        def _update(subtrajectory, stored):
            new_subtrajectory = torch.roll(subtrajectory, shifts=1, dims=1)
            # append to the left
            new_subtrajectory[:, 0, ...] = stored
            subtrajectory.copy_(new_subtrajectory)
            return new_subtrajectory

        return alf.nest.map_structure(
            _update, state.subtrajectory,
            SubTrajectory(
                observation=time_step.observation,
                prev_action=time_step.prev_action))

    def predict_step(self, time_step, state):
        observation, switch_skill = time_step.observation
        first_observation = self._update_state_if_necessary(
            switch_skill, observation, state.first_observation)
        subtrajectory = self._clear_subtrajectory_if_necessary(
            state.subtrajectory, switch_skill)
        return AlgStep(
            state=DiscriminatorState(
                first_observation=first_observation,
                subtrajectory=subtrajectory))

    def rollout_step(self, time_step, state):
        """This function updates the discriminator state."""
        (observation, _, switch_skill, _) = time_step.observation
        first_observation = self._update_state_if_necessary(
            switch_skill, observation, state.first_observation)
        subtrajectory = self._clear_subtrajectory_if_necessary(
            state.subtrajectory, switch_skill)
        return AlgStep(
            state=DiscriminatorState(
                first_observation=first_observation,
                untrans_observation=time_step.untransformed.observation,
                subtrajectory=subtrajectory))

    def train_step(self, exp: Experience, state, trainable=True):
        """This function trains the discriminator or generates intrinsic rewards.

        If ``trainable=True``, then it only generates and returns the pred loss;
        otherwise it only generates rewards with no grad.
        """
        # Discriminator training from its own replay buffer  or
        # Discriminator computing intrinsic rewards for training lower_rl
        untrans_observation, prev_skill, switch_skill, steps = exp.observation

        observation = self._observation_transformer(untrans_observation)
        loss = self._predict_skill_loss(observation, exp.prev_action,
                                        prev_skill, steps, state)

        first_observation = self._update_state_if_necessary(
            switch_skill, observation, state.first_observation)
        subtrajectory = self._clear_subtrajectory_if_necessary(
            state.subtrajectory, switch_skill)
        new_state = DiscriminatorState(
            first_observation=first_observation,
            untrans_observation=untrans_observation,
            subtrajectory=subtrajectory)

        valid_masks = (exp.step_type != StepType.FIRST)
        if self._sparse_reward:
            # Only give intrinsic rewards at the last step of the skill
            valid_masks &= switch_skill
        loss *= valid_masks.to(torch.float32)

        if trainable:
            info = LossInfo(loss=loss, extra=dict(discriminator_loss=loss))
            return AlgStep(state=new_state, info=info)
        else:
            intrinsic_reward = -loss.detach() / self._skill_dim
            return AlgStep(
                state=common.detach(new_state), info=intrinsic_reward)
    
    def calc_loss(self, experience, train_info):
        # This is called for ``train_step(trainable=True)``.
        loss_info = LossInfo(loss=train_info.loss, extra=train_info.extra)
        return loss_info

@gin.configurable
class SkillGenerator(Algorithm):
    """
    Suppose ``num_steps_per_skill=6``:

    ::

        state.steps:

        1 2 3 4 5 6             | 1                  2 3 4 5 6
                  (switch to k')  (boundary discount)

    All the 6 steps in the second half are contributed by :math:`k'`. So for
    a short-term observation memory, we should reset it *after* switching the
    skill within that time step.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 skill_spec,
                 env,
                 config: TrainerConfig,
                 num_steps_per_skill=5,
                 rl_algorithm_cls=SacAlgorithm,
                 rl_mini_batch_size=128,
                 rl_mini_batch_length=2,
                 rl_replay_buffer_length=20000,
                 disc_mini_batch_size=64,
                 disc_mini_batch_length=4,
                 disc_replay_buffer_length=20000,
                 gamma=0.99,
                 optimizer=None,
                 debug_summaries=False,
                 name="SkillGenerator"):
        """
        """
        self._num_steps_per_skill = num_steps_per_skill
        self._observation_spec = observation_spec
        self._action_spec = action_spec
        self._skill_spec = skill_spec

        rl, discriminator = self._create_subalgorithms(
            rl_algorithm_cls, debug_summaries, env, config,
            rl_mini_batch_length, rl_mini_batch_size, rl_replay_buffer_length,
            disc_mini_batch_size, disc_mini_batch_length,
            disc_replay_buffer_length)

        discriminator.set_high_rl(rl)

        train_state_spec = SkillGeneratorState(
            discriminator=discriminator.train_state_spec,  # for discriminator
            skill=self._skill_spec)  # inputs to lower-level

        rollout_state_spec = train_state_spec._replace(
            rl=rl.train_state_spec,  # higher-level policy rollout
            rl_reward=TensorSpec(()),  # higher-level policy replay
            rl_discount=TensorSpec(()),  # higher-level policy replay
            steps=TensorSpec((), dtype='int64'))

        predict_state_spec = train_state_spec._replace(
            rl=rl.predict_state_spec,  # higher-level policy prediction
            steps=TensorSpec((), dtype='int64'),
            discriminator=discriminator.predict_state_spec)

        super().__init__(
            train_state_spec=train_state_spec,
            rollout_state_spec=rollout_state_spec,
            predict_state_spec=predict_state_spec,
            optimizer=optimizer,
            name=name)

        self._gamma = gamma
        self._discriminator = discriminator
        self._rl = rl
        self._rl_train = common.Periodically(
            self._rl.train_from_replay_buffer,
            period=1,
            name="periodic_higher_level")

    def _create_rl_algorithm(self, rl_algorithm_cls, rl_config, env,
                             debug_summaries):
        # pass env and rl_config for creating a replay buffer and metrics
        rl = rl_algorithm_cls(
            observation_spec=self._observation_spec,
            action_spec=self._skill_spec,
            config=rl_config,
            debug_summaries=debug_summaries)
        if rl_config:
            if (rl_config.whole_replay_buffer_training
                    and rl_config.clear_replay_buffer):
                exp_type = "one_time"
            else:
                exp_type = "uniform"
            rl.set_exp_replayer(
                exp_type,
                env.batch_size,
                rl_config.replay_buffer_length,
                prioritized_sampling=False)
        return rl

    def _create_discriminator(self, disc_config, env, num_steps_per_skill,
                              debug_summaries):
        discriminator = Discriminator(
            observation_spec=self._observation_spec,
            action_spec=self._action_spec,
            skill_spec=self._skill_spec,
            config=disc_config,
            num_steps_per_skill=num_steps_per_skill,
            debug_summaries=debug_summaries)
        if disc_config:
            discriminator.set_exp_replayer(
                "uniform", env.batch_size, disc_config.replay_buffer_length,
                prioritized_sampling=False)
        return discriminator

    def _create_subalgorithms(self, rl_algorithm_cls, debug_summaries, env,
                              config, rl_mini_batch_length, rl_mini_batch_size,
                              rl_replay_buffer_length, disc_mini_batch_size,
                              disc_mini_batch_length,
                              disc_replay_buffer_length):

        rl_config = copy.deepcopy(config)  # for training higher-level policy
        disc_config = copy.deepcopy(config)  # for training discriminator
        if config is not None:  # For play, config might be None
            rl_config.replay_buffer_length = rl_replay_buffer_length
            rl_config.initial_collect_steps = config.initial_collect_steps * 5
            rl_config.mini_batch_size = rl_mini_batch_size
            rl_config.mini_batch_length = rl_mini_batch_length

            disc_config.mini_batch_size = disc_mini_batch_size
            disc_config.mini_batch_length = disc_mini_batch_length
            disc_config.replay_buffer_length = disc_replay_buffer_length
            disc_config.initial_collect_steps = config.initial_collect_steps

        rl = self._create_rl_algorithm(rl_algorithm_cls, rl_config, env,
                                       debug_summaries)
        discriminator = self._create_discriminator(
            disc_config, env, self._num_steps_per_skill, debug_summaries)

        return rl, discriminator

    @property
    def num_steps_per_skill(self):
        return self._num_steps_per_skill

    @property
    def output_spec(self):
        return self._skill_spec

    def _trainable_attributes_to_ignore(self):
        # These will train themselves so let the parent algorithm ignore them
        return ["_rl", "_discriminator"]

    def _should_switch_skills(self, time_step: TimeStep, state):
        should_switch_skills = ((state.steps % self._num_steps_per_skill) == 0)
        # is_last is only necessary for `rollout_step` because it marks an
        # episode end in the replay buffer for training the policy `self._rl`.
        return should_switch_skills | time_step.is_first() | time_step.is_last(
        )

    def predict_step(self, time_step: TimeStep, state, epsilon_greedy):
        """This function does one thing, i.e., every ``self._num_steps_per_skill``
        it calls ``self._rl`` to generate new skills.
        """
        switch_skill = self._should_switch_skills(time_step, state)
        discriminator_step = self._discriminator_predict_step(
            time_step, state, switch_skill)

        def _generate_new_skills(time_step, state):
            rl_step = self._rl.predict_step(time_step, state.rl,
                                            epsilon_greedy)
            return SkillGeneratorState(
                skill=rl_step.output,
                steps=torch.zeros_like(state.steps),
                rl=rl_step.state)

        new_state = conditional_update(
            target=state,
            cond=switch_skill,
            func=_generate_new_skills,
            time_step=time_step,
            state=state)

        new_state = new_state._replace(
            steps=new_state.steps + 1, discriminator=discriminator_step.state)

        return AlgStep(
            output=new_state.skill,
            state=new_state,
            info=SkillGeneratorInfo(switch_skill=switch_skill))

    def rollout_step(self, time_step: TimeStep, state):
        r"""This function does three things:

        1. every ``self._num_steps_per_skill`` it calls ``self._rl`` to generate new
           skills.
        2. at the same time writes ``time_step`` to a replay buffer when new skills
           are generated.
        3. call ``rollout_step()`` of the discriminator to write ``time_step``
           to a replay buffer for training

        Regarding accumulating rewards for the higher-level policy. Suppose that
        during an episode we have :math:`H` segments where each segment contains
        :math:`K` steps. Then the objective for the higher-level policy is:

        .. math::

            \begin{array}{ll}
                &\sum_{h=0}^{H-1}(\gamma^K)^h\sum_{t=0}^{K-1}\gamma^t r(s_{t+hK},a_{t+hK})\\
                =&\sum_{h=0}^{H-1}\beta^h R_h\\
            \end{array}

        where :math:`\gamma` is the discount and :math:`r(\cdot)` is the extrinsic
        reward of the original task. Thus :math:`\beta=\gamma^K` should be the
        discount per higher-level time step and :math:`R_h=\sum_{t=0}^{K-1}\gamma^t r(s_{t+hK},a_{t+hK})`
        should the reward per higher-level time step.
        """
        switch_skill = self._should_switch_skills(time_step, state)

        discriminator_step = self._discriminator_rollout_step(
            time_step, state, switch_skill)

        state = state._replace(
            rl_reward=state.rl_reward + state.rl_discount * time_step.reward,
            rl_discount=state.rl_discount * time_step.discount * self._gamma)

        def _generate_new_skills(time_step, state):
            rl_prev_action = state.skill
            # avoid dividing by 0
            #steps = torch.max(
            #    state.steps.to(torch.float32), torch.as_tensor(1.0))
            rl_time_step = time_step._replace(
                reward=state.rl_reward,
                discount=state.rl_discount,
                prev_action=rl_prev_action)

            rl_step = self._rl.rollout_step(rl_time_step, state.rl)

            # store to replay buffer
            self._rl.observe_for_replay(
                make_experience(
                    # ``rl_time_step.observation`` has been transformed!!!
                    rl_time_step._replace(
                        observation=rl_time_step.untransformed.observation),
                    rl_step,
                    state.rl))

            return SkillGeneratorState(
                skill=rl_step.output,
                steps=torch.zeros_like(state.steps),
                discriminator=state.discriminator,
                rl=rl_step.state,
                rl_reward=torch.zeros_like(state.rl_reward),
                rl_discount=torch.ones_like(state.rl_discount))

        new_state = conditional_update(
            target=state,
            cond=switch_skill,
            func=_generate_new_skills,
            time_step=time_step,
            state=state)
        new_state = new_state._replace(
            steps=new_state.steps + 1, discriminator=discriminator_step.state)
        return AlgStep(
            output=new_state.skill,
            state=new_state,
            info=SkillGeneratorInfo(
                skill=new_state.skill,
                steps=new_state.steps,
                switch_skill=switch_skill))

    def train_step(self, exp: Experience, state):
        """This function does the following two things:

        1. take the skill generated during ``rollout_step`` and output it as the
           skill for the current time step.
        2. generate intrinsic rewards using the discriminator (fixed), for training
           the skill-conditioned policy.
        """
        discriminator_step = self._discriminator_train_step(
            exp, state, exp.rollout_info.switch_skill)
        new_state = state._replace(
            skill=exp.rollout_info.skill,
            discriminator=discriminator_step.state)
        return AlgStep(
            output=new_state.skill,
            state=new_state,
            info=SkillGeneratorInfo(reward=discriminator_step.info))

    def _discriminator_rollout_step(self, time_step, state, switch_skill):
        observation = [
            time_step.observation, state.skill, switch_skill,
            state.steps % self._num_steps_per_skill + 1
        ]
        discriminator_step = self._discriminator.rollout_step(
            time_step._replace(observation=observation), state.discriminator)
        # disc_time_step.observation has been transformed!
        disc_time_step = DiscriminatorTimeStep(
            observation=observation,
            prev_action=time_step.prev_action,
            state=state.discriminator,
            env_id=time_step.env_id,
            step_type=time_step.step_type,
            reward=time_step.prev_action.new_zeros((1,)))
        self._discriminator.observe_for_replay(disc_time_step)
        return discriminator_step

    def _discriminator_train_step(self, exp, state, switch_skill):
        return self._discriminator.train_step(
            exp._replace(observation=[
                exp.observation,
                state.skill,
                switch_skill,
                # Potential issue: this steps will be inaccurate if FINAL step
                # comes before num_steps_per_skill
                exp.rollout_info.steps
            ]),
            state.discriminator,
            trainable=False)

    def _discriminator_predict_step(self, time_step, state, switch_skill):
        return self._discriminator.predict_step(
            time_step._replace(
                observation=(time_step.observation, switch_skill)),
            state.discriminator)

    def update_disc_subtrajectory(self, time_step, state):
        return self._discriminator.update_subtrajectory(
            time_step, state.discriminator)

    def calc_loss(self, experience, info: SkillGeneratorInfo):
        return LossInfo()

    def after_train_iter(self, experience, train_info=None):
        with alf.summary.scope(self.name + "_rl"):
            self._rl_train()

        with alf.summary.scope(self._discriminator.name):
            self._discriminator.train_from_replay_buffer()

