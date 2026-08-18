# Prebuilt image for the OpenModelica "validate" job.
#
# Extends the official OpenModelica minimal image and bakes in the extras the
# validation step needs: the Modelica Standard Library (omlibrary) and the
# Python bindings (OMPython, numpy). This removes the per-run apt-get/pip install
# from the validate job.
#
# Rebuild this image only when these tools change (see
# .github/workflows/build-images.yml).

FROM openmodelica/openmodelica:v1.24.0-minimal

# omlibrary provides the Modelica Standard Library, which the -minimal image does
# not ship. python3-pip lets us install the OpenModelica Python bindings.
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip omlibrary && \
    rm -rf /var/lib/apt/lists/*

# --break-system-packages is required on newer Debian bases (PEP 668). Fall back
# to a plain install on images that predate that flag.
RUN pip3 install --break-system-packages OMPython numpy || pip3 install OMPython numpy
