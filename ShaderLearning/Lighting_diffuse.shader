Shader "MyShader/Lighting_diffuse"{
  Properties{
    _BaseColor("MyColor",Color) = (1.0,1.0,1.0,1.0)
  }
  SubShader{
    pass{
      Tags{"LightMode"="ForwardBase"}

      CGPROGRAM
      #pragma vertex Vertex
      #pragma fragment Frag

      #include "Lighting.cginc"

      struct vertexInput{
        float4 vertex : POSITION;
        float3 normal : NORMAL;
      };

      struct vertexOutput{
        float4 pos : SV_POSITION;
        float4 color : COLOR;
      };
 		fixed4 _BaseColor;

      vertexOutput Vertex(vertexInput v){
        vertexOutput o;
        o.pos = UnityObjectToClipPos(v.vertex);

        fixed3 worldNormal = UnityObjectToWorldNormal(v.normal);
        fixed3 worldNormalStd = normalize(worldNormal);
        fixed3 worldLightDir = normalize(_WorldSpaceLightPos0.xyz);

        fixed3 diffuse = _LightColor0.xyz * max(0,dot(worldNormal,worldLightDir));
        diffuse=diffuse*_BaseColor.xyz;

        o.color = fixed4(diffuse,1.0);
        return o;
      }

      fixed4 Frag(vertexOutput i):SV_TARGET{
        /*不需要计算任何东西*/
        return i.color;
      }
	   ENDCG
    }
  }
}