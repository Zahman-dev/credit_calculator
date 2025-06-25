# Base on the same python version for consistency
FROM python:3.10.17-slim

# Set working directory
WORKDIR /home/jovyan/work

# Install JupyterLab and common data science libraries
RUN pip install --no-cache-dir jupyterlab notebook pandas matplotlib scikit-learn seaborn

# Expose port
EXPOSE 8888

# Set user and permissions
RUN useradd -m -s /bin/bash -N -u 1000 jovyan && \
    chown -R jovyan:users /home/jovyan

USER jovyan

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''"] 