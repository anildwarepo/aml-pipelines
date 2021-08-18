import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.pipeline.core import PipelineData
from azureml.pipeline.core import PipelineParameter
from azureml.pipeline.steps import EstimatorStep
from azureml.train.dnn import PyTorch
from azureml.core import ScriptRunConfig
from azureml.pipeline.steps import CommandStep
from azureml.core import Environment

def train_step(train_dir, valid_dir, compute_target, work_space):
    '''
    This step will fine-tune a RESNET-18 model on our dataset using PyTorch. 
    It will use the corresponding input image directories as training and validation data.

    :param train_dir: The reference to the directory containing the training data
    :type train_dir: DataReference
    :param valid_dir: The reference to the directory containing the validation data
    :type valid_dir: DataReference
    :param compute_target: The compute target to run the step on
    :type compute_target: ComputeTarget
    
    :return: The preprocess step, step outputs dictionary (keys: model_dir)
    :rtype: EstimatorStep, dict
    '''

    num_epochs = PipelineParameter(name='num_epochs', default_value=25)
    batch_size = PipelineParameter(name='batch_size', default_value=16)
    learning_rate = PipelineParameter(name='learning_rate', default_value=0.001)
    momentum = PipelineParameter(name='momentum', default_value=0.9)
    project_folder = './scripts'
    model_dir = PipelineData(
        name='model_dir', 
        pipeline_output_name='model_dir',
        datastore=train_dir.datastore,
        output_mode='mount',
        is_directory=True)

    outputs = [model_dir]
    outputs_map = { 'model_dir': model_dir }


    #curated_env_name = 'AzureML-PyTorch-1.6-GPU'
    #pytorch_env = Environment.get(workspace=work_space, name=curated_env_name)
    pytorch_env = Environment.from_conda_specification(name='pytorch-1.6-gpu', file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'conda_dependencies.yml'))

    # Specify a GPU base image
    pytorch_env.docker.enabled = True
    pytorch_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'

    train_config = ScriptRunConfig(source_directory=os.path.dirname(os.path.abspath(__file__)),
                      command=['python', 'train.py', '--output_dir', './outputs',  '--train_dir', train_dir, 
                                      '--valid_dir', valid_dir, 
                                      '--output_dir', model_dir, 
                                      '--num_epochs', num_epochs, 
                                      '--batch_size', batch_size,
                                      '--learning_rate', learning_rate, 
                                      '--momentum', momentum
                                    ],
                      compute_target=compute_target,
                      environment=pytorch_env)
    
    train_step = CommandStep(name='train-step', inputs=[train_dir, valid_dir],
                      outputs=outputs, runconfig=train_config)
    
    return train_step, outputs_map
    
