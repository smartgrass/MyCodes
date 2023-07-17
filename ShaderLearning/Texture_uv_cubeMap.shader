Shader "Unlit/Texture_uv_cubeMap"
{
    Properties
    {
		_BaseColor("MyColor",Color) = (1.0,1.0,1.0,1.0)
		_Gloss("Gloss",Range(8,100)) = 20.0
		_Intensity ("_Intensity", Range(0,1)) = 0.5
        _MainTex("Main Texture",2D) = "white"{}
		_MainTex_ST("offset",Vector) = (1,1,0,0)
		_ReflectAmount("Reflection Amount",Range(0,1)) = 1
		_RefractRatio("Refract Ratio",Range(0.1,1)) = 0.5
		_Cubemap("Reflection Cubemap",Cube) = "skybox"{}
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
			#include "AutoLight.cginc"

            struct inputData{
                float4 vertex : POSITION;
                float3 normal : NORMAL;
				float4 texcoord : TEXCOORD0;
            };

            struct outputData{
                float4 pos : SV_POSITION;
                float3 worldNormal : TEXCOORD0;
				float2 uv : TEXCOORD1;
				float3 worldPos : TEXCOORD2;   //需要模型的世界坐标位置，才可以计算视角方向
            };

            fixed4 _BaseColor;
            float _Gloss;
			float _Intensity;
			sampler2D _MainTex;
			float4 _MainTex_ST;
			float _ReflectAmount;
            samplerCUBE _Cubemap;
			float _RefractRatio;

            outputData vert (inputData v)
            {
                outputData o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                o.worldPos = mul(unity_ObjectToWorld,v.vertex).xyz;    //模型坐标转世界坐标,float4->float3
				o.uv = v.texcoord.xy *_MainTex_ST.xy + _MainTex_ST.wz;
				return o;
            }

			fixed4 frag(outputData i):SV_TARGET{

				//标准化
                fixed3 worldNormal = normalize(i.worldNormal);
                fixed3 worldLightDir = normalize(_WorldSpaceLightPos0.xyz);

				//reflect(光源，法线)
                fixed3 reflectDir = reflect(-worldLightDir,worldNormal);

				//相机
                fixed3 viewDir = normalize(UnityWorldSpaceViewDir(i.worldPos));

				float3 albedo = tex2D(_MainTex,i.uv).xyz * _BaseColor.xyz;

                fixed3 ambient = UNITY_LIGHTMODEL_AMBIENT.xyz * albedo;    //先获取环境光

				//saturate 相当于max加clam(0,1)
                fixed3 diffuse = _LightColor0.xyz * albedo * saturate(dot(worldNormal,worldLightDir));

   				//利用Phong模型中的高光公式来计算高光
                fixed3 specular = _LightColor0.xyz * pow(saturate(dot(viewDir,reflectDir)),_Gloss);

				fixed3 reflectViewDir = reflect(-viewDir,worldNormal);
				// fixed3 reflectViewDir = refract(-viewDir,worldNormal,_RefractRatio);

			 	fixed3 reflection = texCUBE(_Cubemap,reflectViewDir).xyz;

				fixed3 color = specular+ ambient + lerp(diffuse,reflection,_ReflectAmount);

                return fixed4(color,1.0);
            }

            ENDCG
        }
    }
}
