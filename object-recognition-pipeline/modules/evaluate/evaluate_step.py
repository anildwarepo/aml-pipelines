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

def evaluate_step(model_dir, test_dir, compute_target, work_space):
    '''
    This step evaluates the trained model on the testing data and outputs the accuracy.

    :param model_dir: The reference to the directory containing the trained model
    :type model_dir: DataReference
    :param test_dir: The reference to the directory containing the testing data
    :type test_dir: DataReference
    :param compute_target: The compute target to run the step on
    :type compute_target: ComputeTarget
    
    :return: The preprocess step, step outputs dictionary (keys: accuracy_file)
    :rtype: EstimatorStep, dict
    '''

    accuracy_file = PipelineData(
        name='accuracy_file', 
        pipeline_output_name='accuracy_file',
        datastore=test_dir.datastore,
        output_mode='mount',
        is_directory=False)

    outputs = [accuracy_file]
    outputs_map = { 'accuracy_file': accuracy_file }
    
    curated_env_name = 'AzureML-PyTorch-1.6-GPU'
    pytorch_env = Environment.get(workspace=work_space, name=curated_env_name)

    #pytorch_env = Environment.from_conda_specification(name='pytorch-1.6-gpu', file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'conda_dependencies.yml'))

    # Specify a GPU base image
    #pytorch_env.docker.enabled = True
    #pytorch_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04'

    estimator_config = ScriptRunConfig(source_directory=os.path.dirname(os.path.abspath(__file__)),
                      command=['python', 'evaluate.py',  '--test_dir', test_dir, 
                                '--model_dir', model_dir, 
                                '--accuracy_file', accuracy_file
                            ],
                      compute_target=compute_target,
                      environment=pytorch_env)
    
    evaulate_step = CommandStep(name='evaulate-step', inputs=[model_dir, test_dir],
        outputs=outputs, runconfig=estimator_config)

   

    return evaulate_step, outputs_map
