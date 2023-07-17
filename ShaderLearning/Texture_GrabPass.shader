Shader "MyShader/Texture_GrabPass"
{
    Properties
    {
		_BaseColor("Base Color",Color) = (1.0,1.0,1.0,1.0)
		_MainTex("Main Texture",2D) = "white"{}
    }
    SubShader
    {
        Tags{"Queue"="Transparent" "RenderType"="Opaque"}
		GrabPass{"_GrabPassTexture"}
		//GrabPass会将屏幕内容输出到指定纹理
        Pass
        {
			Tags{"LightMode"="Always"}

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
				float4 uv : TEXCOORD0;
            };

			sampler2D _GrabPassTexture;
			sampler2D _MainTex;
			fixed4 _BaseColor;

            outputData vert(inputData i)
            {
                outputData o;
                o.pos = UnityObjectToClipPos(i.vertex);
				o.uv.xy = i.texcoord.xy;

				fixed4 screenPos = ComputeGrabScreenPos(o.pos);
				o.uv.zw = screenPos.xy/screenPos.w;

			/*或者手动计算
				//齐次裁剪空间的坐标为(x, y, z, w)，变换到NDC空间的坐标即为(x/w, y/w, z/w)
				//NDC空间x,y的范围在[-1,1], 需要将其映射到[0,1], (p+1)/2
				fixed2 ndc = o.pos.xy/o.pos.w;
				ndc.y *= -1;
				o.uv.zw = (ndc+1)/2;
			*/

				return o;
            }

			fixed4 frag(outputData o):SV_TARGET{

				fixed3 albedo =tex2D(_MainTex,o.uv.xy);
				fixed3 color = tex2D(_GrabPassTexture,o.uv.zw).xyz *_BaseColor *albedo;
                return fixed4(color,1.0);
            }

			/*ComputeGrabScreenPos 源码
			ComputeGrabScreenPos(fixed4 pos){
				#ifdef UNITY_UV_STARTS_AT_TOP
					float scale = -1.0;
				#else
					float scale = 1.0;
				#endif

				float4 _currentPos = pos * 0.5;
				_currentPos.xy = float2(_currentPos.x,_currentPos.y * scale) + _currentPos.w;
				_currentPos.zw = pos.zw;
				return _currentPos;
			}
			*/

            ENDCG
        }
    }
}
