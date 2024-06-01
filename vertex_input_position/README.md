# Vertex Input Position

If you have a shader such as

```glsl
layout (location = 0) in vec2 inUV;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec3 inPos;

void main() {
	gl_Position = ubo.projection * ubo.model * vec4(inPos.xyz, 1.0);
}
```

This pass will help detect which vertex input `Location` was used to write the `Position` built-in

In the above example, because `inPos` is used, it will let us know `Location 2` was involved