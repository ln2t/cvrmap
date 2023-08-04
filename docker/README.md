# Notes to build this docker image

If you need to build the image yourself, run

```
docker build . -t sometag --no-cache
```

We use the `--no-cache` option to ensure it fetches the last version of the git repo.

Note that in principle the image has been pushed to docker hub and can be installed directely using

```
docker pull arovai/cvrmap:VERSION
```

where `VERSION` must be replaced by the version you wish to install (e.g. `v1.0`).
