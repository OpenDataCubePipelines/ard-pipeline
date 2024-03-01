

Run `./create-module.sh` to create an NCI module.

It will show you the variables it's using and ask for confirmation.

You can override any variables by setting them in bash before calling.

```
❯ ./create-module.sh
##########################
module_dir = /g/data/up71/modules
swfo_version= swfo-0.0.2
gost_version = gost-0.0.3
modtran_version = 6.0.1
DATACUBE_CONFIG_PATH = /g/data/v10/public/modules/dea/20221025/datacube.conf
##########################

Packaging "ard-pipeline 20230823-1426" to "/g/data/up71/modules/ard-pipeline/20230823-1426"

Continue? [y/N]
```

The script `./create-s2-module.sh` is an example of overriding variables for
an S2 module (a specific fmask, and nbart-only)

The script `./create-dev-module.sh` gives an example of calling it with a different module location.

After loading the your new module, you can run `../check-environment.sh` to check that dependencies
and native modules load correctly.
