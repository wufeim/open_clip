## Differentiable Rendering Example

### Example: NeMo

NeMo requires differentiable rendering because of the analysis-by-synthesis design -- updating pose predicting based on the feature reconstruction loss.

1. Here `C` and `theta` contains pose parameters: [link](https://github.com/wufeim/NeMo/blob/master/nemo/models/solve_pose.py#L224).
2. We update `C` and `theta` based on feature reconstruction loss: [link](https://github.com/wufeim/NeMo/blob/master/nemo/models/solve_pose.py#L248).
3. This `inter_module` is a PyTorch3D module that differentiably render feature maps based on `C` and `theta`: [link](https://github.com/wufeim/NeMo/blob/master/nemo/models/solve_pose.py#L259).
4. This is the feature reconstruction loss: [link](https://github.com/wufeim/NeMo/blob/master/nemo/models/solve_pose.py#L271).
5. Here `inter_module` is a interpolation module that based on PyTorch3D implementation and initializes with the mesh `xvert` and `xface`, mesh features `self.feature_bank`, and other rasterization settings: [link](https://github.com/wufeim/NeMo/blob/master/nemo/models/nemo.py#L281).
