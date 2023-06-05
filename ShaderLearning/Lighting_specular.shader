Shader "MyShader/Lighting_specular"
{
    Properties
    {
		_BaseColor("MyColor",Color) = (1.0,1.0,1.0,1.0)
		_Gloss("Gloss",Range(8,100)) = 20.0
		_Intensity ("_Intensity", Range(0,1)) = 0.5
        _MainTex ("Texture", 2D) = "white" {}
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
            };

            struct outputData{
                float4 pos : SV_POSITION;
                float3 worldNormal : TEXCOORD0;
                float3 worldPos : TEXCOORD1;   //需要模型的世界坐标位置，才可以计算视角方向
            };

            fixed4 _BaseColor;
            float _Gloss;
			float _Intensity;

            outputData vert (inputData v)
            {
                outputData o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                o.worldPos = mul(unity_ObjectToWorld,v.vertex).xyz;    //模型坐标转世界坐标,float4->float3
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

                fixed3 ambient = UNITY_LIGHTMODEL_AMBIENT.xyz * _BaseColor.xyz;    //先获取环境光
				//saturate 相当于max加clam(0,1)
                fixed3 diffuse = _LightColor0.xyz * _BaseColor.xyz * saturate(dot(worldNormal,worldLightDir));

   				//利用Phong模型中的高光公式来计算高光
                fixed3 specular = _LightColor0.xyz * pow(saturate(dot(viewDir,reflectDir)),_Gloss);


                return fixed4(specular + ambient + diffuse,1.0);
            }

            ENDCG
        }
    }
}
