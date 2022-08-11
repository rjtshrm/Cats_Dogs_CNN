import uuid
import kfp
import kfp.dsl as dsl
from kfp.dsl import ContainerOp, component
import kubernetes.client as k8s_client
import kfp.compiler as compiler
from kubernetes.client import V1EnvVar


class MLPipeline:

    def __init__(self, exp_name, host_url, namespace, pvc_name):
        self.exp_name = exp_name
        self.client = kfp.Client(host=host_url, namespace="anonymous")
        print(self.client.list_experiments(namespace="anonymous"))
        self.pvc_name = pvc_name


    def data_generator(self) -> ContainerOp:
        op = ContainerOp(
            name="data_generagtor",
            image="rjtshrm/data_generator:latest",
            arguments=["--data_dir", "/tmp/"],
            container_kwargs={'env': [V1EnvVar('EXP_NAME', self.exp_name)]}
        ).add_volume(k8s_client.V1Volume(
            persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=self.pvc_name),
            name='kubeflowpvc'))\
            .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/tmp/', name='kubeflowpvc')).set_image_pull_policy('Always')

        return op

    def trainer(self) -> ContainerOp:
        op = ContainerOp(
            name="model_trainer",
            image="rjtshrm/trainer:latest",
            arguments=["--mlflow_host", "http://20.28.195.42/mlflow/", "--ds_path", "/tmp/Cat_Dog", "--epoch", 1],
            container_kwargs={'env': [V1EnvVar('EXP_NAME', self.exp_name)]}
        ).add_volume(k8s_client.V1Volume(
            persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(claim_name=self.pvc_name),
            name='kubeflowpvc'))\
            .add_volume_mount(k8s_client.V1VolumeMount(mount_path='/tmp/', name='kubeflowpvc')).set_image_pull_policy('Always')

        return op

    @dsl.pipeline(name="Classification pipeline",
                  description="A cat and dog Classification pipeline on Kubefllow",)
    def pipeline(self,):
        data_generator_step = self.data_generator()
        data_preprocessor_step = self.trainer().after(data_generator_step)


    def pipeline_start(self):
        self.client.create_run_from_pipeline_func(
            self.pipeline, arguments={}
        )


if __name__ == "__main__":
    mlp = MLPipeline(exp_name=str(uuid.uuid4()), host_url="http://20.28.195.42/pipeline", namespace="anonymous", pvc_name="kubeflow-pvc")
    mlp.pipeline_start()