docker build . -t powersgd-async
docker tag powersgd-async ic-registry.epfl.ch/mlo/powersgd-async
docker push ic-registry.epfl.ch/mlo/powersgd-async
kubectl apply -f runai.yml