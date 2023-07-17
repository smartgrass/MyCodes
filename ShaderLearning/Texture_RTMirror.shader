Shader "Unlit/Texture_RTMirror"
{
    Properties
    {
        _MainTex("Main Texture",2D) = "white"{}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

			#include "Lighting.cginc"
            #include "UnityCG.cginc"

            struct inputData{
                float4 vertex : POSITION;
                float3 normal : NORMAL;
				float4 texcoord : TEXCOORD0;
            };

            struct outputData{
                float4 pos : SV_POSITION;
				float2 uv : TEXCOORD1;
            };

			sampler2D _MainTex;

            outputData vert (inputData v)
            {
                outputData o;
                o.pos = UnityObjectToClipPos(v.vertex);
				//由于是镜像,所以取反
				o.uv =1-(v.texcoord.xy);
				return o;
            }

			fixed4 frag(outputData i):SV_TARGET{
                return fixed4(tex2D(_MainTex,i.uv).xyz,1.0);
            }

            ENDCG
        }
    }
}
