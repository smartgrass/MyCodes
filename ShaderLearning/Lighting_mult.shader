// Upgrade NOTE: replaced '_LightMatrix0' with 'unity_WorldToLight'

Shader "MyShader/Lighting_mult"
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
			Tags{"LightMode"="ForwardBase"}
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

			#include "Lighting.cginc"
            #include "UnityCG.cginc"
			#pragma multi_compile_fwdbase  //一定要有这个编译指令

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


        Pass
        {
			Tags{"LightMode"="ForwardAdd"}
			Blend One One //开启混合，设置线性混合
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

			#include "Lighting.cginc"
            #include "UnityCG.cginc"
			#include "AutoLight.cginc"
			#pragma multi_compile_fwdadd   //一定要有这个编译指令

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

				///计算光照方向
				#ifdef USING_DIRECTIONAL_LIGHT
							//如果是平行光
					fixed3 lightDir = normalize(_WorldSpaceLightPos0.xyz);
				#else
					//不是平行光
					fixed3 lightDir = normalize(_WorldSpaceLightPos0.xyz - i.worldPos);
				#endif

				//reflect(光源，法线)
                fixed3 reflectDir = reflect(-lightDir,worldNormal);

				//相机
                fixed3 viewDir = normalize(UnityWorldSpaceViewDir(i.worldPos));

                // fixed3 ambient 环境光前个pass算过一次就够了

                fixed3 diffuse = _LightColor0.xyz * _BaseColor.xyz * saturate(dot(worldNormal,lightDir));

   				//利用Phong模型中的高光公式来计算高光
                fixed3 specular = _LightColor0.xyz * pow(saturate(dot(viewDir,reflectDir)),_Gloss);

				///计算光照衰减
				#ifdef USING_DIRECTIONAL_LIGHT
					fixed atten = 1.0;   // 平行光不衰减
				#else
					#if defined(POINT)
						//点光源
						//计算顶点在 光源空间 的位置
						float3 lightCoord = mul(unity_WorldToLight,float4(i.worldPos,1)).xyz;
						fixed atten = tex2D(_LightTexture0,dot(lightCoord,lightCoord).rr).UNITY_ATTEN_CHANNEL;
					#elif defined(SPOT)
						//聚光灯,较为复杂,跳过..
						float4 lightCoord = mul(unity_WorldToLight,float4(i.worldPos,1));
						fixed atten=(lightCoord.z>0)*tex2D(_LightTexture0,lightCoord.xy/lightCoord.w+0.5).w*tex2D(_LightTextureB0,dot(lightCoord,lightCoord).rr).UNITY_ATTEN_CHANNEL;
					#endif
				#endif


                return fixed4((specular + diffuse)*atten,1.0);
            }

            ENDCG
        }

    }
}
