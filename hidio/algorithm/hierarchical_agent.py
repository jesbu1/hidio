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
"""Agent for integrating multiple algorithms."""

import gin

import torch
from torch import nn

import alf
from alf.algorithms.sac_algorithm import SacAlgorithm
from alf.algorithms.agent_helpers import AgentHelper
from alf.algorithms.config import TrainerConfig
from .skill_generator import SkillGenerator, SubTrajectory
from alf.algorithms.on_policy_algorithm import OnPolicyAlgorithm
from alf.data_structures import AlgStep, Experience
from alf.data_structures import TimeStep, namedtuple
from alf.nest.utils import transform_nest
from alf.utils import math_ops
from alf.data_structures import StepType
from alf.tensor_specs import BoundedTensorSpec, TensorSpec
from alf.networks.preprocessors import EmbeddingPreprocessor
from alf.utils.conditional_ops import conditional_update

AgentState = namedtuple(
    "AgentState", ["rl", "skill_generator"], default_value=())

AgentInfo = namedtuple(
    "AgentInfo", ["rl", "skill_generator", "skill_discount"], default_value=())


@gin.configurable
def get_low_rl_input_spec(observation_spec, action_spec, num_steps_per_skill,
                          skill_spec):
    assert observation_spec.ndim == 1 and action_spec.ndim == 1
    concat_observation_spec = TensorSpec(
        (num_steps_per_skill * observation_spec.shape[0], ))
    concat_action_spec = TensorSpec(
        (num_steps_per_skill * action_spec.shape[0], ))
    traj_spec = SubTrajectory(
        observation=concat_observation_spec, prev_action=concat_action_spec)
    step_spec = step_spec = BoundedTensorSpec(
        shape=(), maximum=num_steps_per_skill, dtype='int64')
    return alf.nest.flatten(traj_spec) + [step_spec, skill_spec]


@gin.configurable
def get_low_rl_input_preprocessors(low_rl_input_specs, embedding_dim):
    return alf.nest.map_structure(
        lambda spec: EmbeddingPreprocessor(
            input_tensor_spec=spec, embedding_dim=embedding_dim),
        low_rl_input_specs)


@gin.configurable
class HierarchicalAgent(OnPolicyAlgorithm):
    """Hierarchical Agent

    Higher-level policy proposes skills for lower-level policy to executes. The
    rewards for the former is the extrinsic rewards, while the rewards for the
    latter is the negative of a skill discrimination loss.
    """

    def __init__(self,
                 observation_spec,
                 action_spec,
                 skill_spec,
                 env=None,
                 config: TrainerConfig = None,
                 skill_generator_cls=SkillGenerator,
                 rl_algorithm_cls=SacAlgorithm,
                 skill_boundary_discount=1.0,
                 optimizer=None,
                 observation_transformer=math_ops.identity,
                 exp_reward_relabeling=True,
                 debug_summaries=False,
                 name="AgentAlgorithm"):
        """Create an Agent

        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            env (Environment): The environment to interact with. `env` is a
                batched environment, which means that it runs multiple
                simulations simultaneously. Running multiple environments in
                parallel is crucial to on-policy algorithms as it increases the
                diversity of data and decreases temporal correlation. `env` only
                needs to be provided to the root `Algorithm`.
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs `train_iter()` by
                itself.
            rl_algorithm_cls (type): The algorithm class for learning the policy.
            skill_generator (Algorithm): an algorithm with output a goal vector
            optimizer (tf.optimizers.Optimizer): The optimizer for training
            observation_transformer (Callable | list[Callable]): transformation(s)
                applied to `time_step.observation`
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
            """
        print("Skill spec:", skill_spec)

        agent_helper = AgentHelper(AgentState)

        ## 1. goal generator
        skill_generator = skill_generator_cls(
            observation_spec=observation_spec,
            action_spec=action_spec,
            skill_spec=skill_spec,
            env=env,
            config=config,
            debug_summaries=debug_summaries)
        agent_helper.register_algorithm(skill_generator, "skill_generator")

        rl_observation_spec = get_low_rl_input_spec(
            observation_spec, action_spec, skill_generator.num_steps_per_skill,
            skill_generator.output_spec)

        ## 2. rl algorithm
        # Currently the hierarchical agent only supports SAC.
        rl_algorithm = rl_algorithm_cls(
            observation_spec=rl_observation_spec,
            action_spec=action_spec,
            config=config,  # set use_rollout_state
            debug_summaries=debug_summaries)
        agent_helper.register_algorithm(rl_algorithm, "rl")
        # Whether the agent is on-policy or not depends on its rl algorithm.
        self._is_on_policy = rl_algorithm.is_on_policy()
        self._skill_boundary_discount = skill_boundary_discount
        self._exp_reward_relabeling = exp_reward_relabeling

        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            optimizer=optimizer,
            env=env,
            config=config,
            debug_summaries=debug_summaries,
            name=name,
            **agent_helper.state_specs())

        self._rl_algorithm = rl_algorithm
        self._skill_generator = skill_generator
        self._agent_helper = agent_helper
        self._observation_transformer = observation_transformer
        self._num_steps_per_skill = skill_generator.num_steps_per_skill

    def is_on_policy(self):
        return self._is_on_policy

    def _make_low_level_observation(self, subtrajectory, skill, switch_skill,
                                    steps, updated_first_observation):
        r"""Given the skill generator's output, this function makes the
        skill-conditioned observation for the lower-level policy. Both observation
        and action are a stacking of recent states, with the most recent one appearing
        at index=0.

        X: first observation of a skill
        X': first observation of the next skill
        O: middle observation of a skill
        _: zero

        num_steps_per_skill=3:

            subtrajectory (discriminator)    low_rl_observation
            O _ _                        ->  O X _   (steps==2)
            O O _                        ->  O O X   (steps==3)
            X'O O                        ->  X'_ _   (steps==1,switch_skill==True)

        The same applies to action except that there is no ``first_observation``.
        """
        #print("steps:", steps)

        subtrajectory.observation[torch.arange(updated_first_observation.
                                               shape[0]).long(), steps -
                                  1] = updated_first_observation

        def _zero(subtrajectory):
            subtrajectory.prev_action.fill_(0.)
            # When switch_skill is because of FINAL steps, filling
            # 0s might have issues if the FINAL step comes before num_steps_per_skill.
            # But since RL algorithms don't train FINAL steps, for now we'll leave
            # it like this for simplicity.
            subtrajectory.observation[:, 1:, ...] = 0.
            return subtrajectory

        subtrajectory = conditional_update(
            target=subtrajectory,
            cond=switch_skill,
            func=_zero,
            subtrajectory=subtrajectory)
        subtrajectory = alf.nest.map_structure(
            lambda traj: traj.reshape(traj.shape[0], -1), subtrajectory)

        #print("switch_skill:", switch_skill)

        low_rl_observation = (alf.nest.flatten(subtrajectory) +
                              [self._num_steps_per_skill - steps, skill])
        #print("low_rl_observation:", low_rl_observation)
        return low_rl_observation

    def predict_step(self, time_step: TimeStep, state: AgentState,
                     epsilon_greedy):
        """Predict for one step."""
        new_state = AgentState()

        time_step = transform_nest(time_step, "observation",
                                   self._observation_transformer)

        subtrajectory = self._skill_generator.update_disc_subtrajectory(
            time_step, state.skill_generator)

        skill_step = self._skill_generator.predict_step(
            time_step, state.skill_generator, epsilon_greedy)
        new_state = new_state._replace(skill_generator=skill_step.state)

        observation = self._make_low_level_observation(
            subtrajectory, skill_step.output, skill_step.info.switch_skill,
            skill_step.state.steps,
            skill_step.state.discriminator.first_observation)

        rl_step = self._rl_algorithm.predict_step(
            time_step._replace(observation=observation), state.rl,
            epsilon_greedy)
        new_state = new_state._replace(rl=rl_step.state)

        return AlgStep(output=rl_step.output, state=new_state)

    def rollout_step(self, time_step: TimeStep, state: AgentState):
        """Rollout for one step."""
        new_state = AgentState()
        info = AgentInfo()

        time_step = transform_nest(time_step, "observation",
                                   self._observation_transformer)

        subtrajectory = self._skill_generator.update_disc_subtrajectory(
            time_step, state.skill_generator)

        skill_step = self._skill_generator.rollout_step(
            time_step, state.skill_generator)
        new_state = new_state._replace(skill_generator=skill_step.state)
        info = info._replace(skill_generator=skill_step.info)

        #print("step_type:", time_step.step_type)
        observation = self._make_low_level_observation(
            subtrajectory, skill_step.output, skill_step.info.switch_skill,
            skill_step.state.steps,
            skill_step.state.discriminator.first_observation)

        rl_step = self._rl_algorithm.rollout_step(
            time_step._replace(observation=observation), state.rl)
        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        skill_discount = ((
            (skill_step.state.steps == 1)
            & (time_step.step_type != StepType.LAST)).to(torch.float32) *
                          (1 - self._skill_boundary_discount))
        info = info._replace(skill_discount=1 - skill_discount)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def train_step(self, exp: Experience, state):
        new_state = AgentState()
        info = AgentInfo()

        skill_generator_info = exp.rollout_info.skill_generator

        subtrajectory = self._skill_generator.update_disc_subtrajectory(
            exp, state.skill_generator)
        skill_step = self._skill_generator.train_step(
            exp._replace(rollout_info=skill_generator_info),
            state.skill_generator)
        info = info._replace(skill_generator=skill_step.info)
        new_state = new_state._replace(skill_generator=skill_step.state)

        exp = transform_nest(exp, "observation", self._observation_transformer)

        observation = self._make_low_level_observation(
            subtrajectory, skill_step.output,
            skill_generator_info.switch_skill, skill_generator_info.steps,
            skill_step.state.discriminator.first_observation)

        rl_step = self._rl_algorithm.train_step(
            exp._replace(
                observation=observation, rollout_info=exp.rollout_info.rl),
            state.rl)

        new_state = new_state._replace(rl=rl_step.state)
        info = info._replace(rl=rl_step.info)

        return AlgStep(output=rl_step.output, state=new_state, info=info)

    def calc_loss(self, experience, train_info: AgentInfo):
        """Calculate loss."""
        if self._exp_reward_relabeling:
            # replace reward in `experience` with the freshly
            # computed intrinsic rewards by the goal generator during `train_step`.
            # For PPOAlgorithm, this relabeling doesn't affect the training because
            # its advantages are computed only once in ``preprocess_experience()``
            # before updates.
            skill_info = train_info.skill_generator
            experience = experience._replace(reward=skill_info.reward)

        self.summarize_reward("training_reward/low_level_intrinsic",
                              experience.reward)

        return self._agent_helper.accumulate_loss_info(
            [self._rl_algorithm, self._skill_generator], experience,
            train_info)

    def after_update(self, experience, train_info: AgentInfo):
        self._agent_helper.after_update(
            [self._rl_algorithm, self._skill_generator], experience,
            train_info)

    def after_train_iter(self, experience, train_info: AgentInfo = None):
        self._agent_helper.after_train_iter(
            [self._rl_algorithm, self._skill_generator], experience,
            train_info)

    def preprocess_experience(self, exp: Experience):
        self.summarize_reward("training_reward/extrinsic", exp.reward)
        # relabel exp with intrinsic rewards by goal generator
        skill_rollout_info = exp.rollout_info.skill_generator
        new_exp = self._rl_algorithm.preprocess_experience(
            exp._replace(
                reward=skill_rollout_info.reward,
                discount=exp.discount * exp.rollout_info.skill_discount,
                rollout_info=exp.rollout_info.rl))
        return new_exp._replace(
            rollout_info=exp.rollout_info._replace(rl=new_exp.rollout_info))