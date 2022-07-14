import itertools
import subprocess
import tempfile
import os
from typing import Iterable

launch_settings = {
    "optimizers": [
        #["sgd"], 
        #["async-sgd"], 
        #["powersgd-async"], 
    ],

    "start-lr": [0.2]
}


def launch():
    os.system("docker build . -t powersgd-async")
    os.system("docker tag powersgd-async ic-registry.epfl.ch/mlo/powersgd-async")
    os.system("docker push ic-registry.epfl.ch/mlo/powersgd-async")

    for id, args_values in enumerate(itertools.product(*launch_settings.values())):
        kwargs = {k: v for k, v in zip(launch_settings.keys(), args_values)}
        print(kwargs)
        create(id + 1, **kwargs)

def create(id, **kwargs):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    f.write(template(id, **kwargs))
    f.close()
    subprocess.call(["kubectl", "create", "-f", f.name])
    os.unlink(f.name)

def format_value(value):
  if isinstance(value, Iterable):
    return ' '.join(value)
  return value

def template(id, **kwargs):
    args = [f'"--{name}={format_value(value)}"' for name, value in kwargs.items()]
    str_args = ', '.join(args)

    return f"""
    apiVersion: run.ai/v1
    kind: RunaiJob
    metadata:
      name: oyounis-async-{id}
      labels:
        user: oyounis
    spec:
      template:
        metadata:
          labels:
            user: oyounis
        spec:	
          hostIPC: true
          schedulerName: runai-scheduler
          restartPolicy: Never
          securityContext:
            runAsUser: 252255
            runAsGroup: 30034
            fsGroup: 30034
          containers:
          - name: container-name
            image: ic-registry.epfl.ch/mlo/powersgd-async
            workingDir : /home/oyounis
            command: [
              "/bin/bash",
              "entrypoint.sh"
              ]
            args: [{str_args}]
            resources:
              limits:
                nvidia.com/gpu: 1
            volumeMounts:
              - mountPath: /mlodata1
                name: mlodata1
          volumes:
            - name: mlodata1
              persistentVolumeClaim:
                claimName: runai-pv-mlodata1

    """

if __name__ == "__main__":
    launch()