# Using Soft Labels for Goal Prediction

This code is modified from the official implementation of [DenseTNT](https://github.com/Tsinghua-MARS-Lab/DenseTNT).

## Soft Labels for Dense Goals

Inspired by the artificial potential field (APF) approach for path planning, for each goal candidate, we compute the target attraction and reference path attraction based on their distance to the goal candidate. A softmax function is applied to the sum of these "attractive forces" to serve as a soft label for dense goal classification (or retrieval).

**Remark.** The reference path is a polyline crossing through the target and looking similar to the centerline of the closest lane. The distance from a goal candidate to a reference path is computed by the minimum distance from the candidate to a point on the path. By considering attractions from this reference path, we implicitly incorporate the prior knowledge that as long as the model predicts a goal with a similar intention, the trajectory decoder (e.g. frenet-based) would achieve similar results.

Examples are illustrated below:
<p align="center">
  <img src="./figures/dense_goal_heatmap_example-1.png" alt="dense_goal_example-1.png" width="300"/>
  <img src="./figures/dense_goal_heatmap_example-2.png" alt="dense_goal_example-1.png" width="300"/>
  <img src="./figures/dense_goal_heatmap_example-3.png" alt="dense_goal_example-1.png" width="300"/>
</p>
