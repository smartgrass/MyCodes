Shader "Unlit/Texture_bump"
{
    Properties
    {
		_BaseColor("MyColor",Color) = (1.0,1.0,1.0,1.0)
		_Gloss("Gloss",Range(8,100)) = 20.0
		_Intensity ("_Intensity", Range(0,1)) = 0.5
        _MainTex("Main Texture",2D) = "white"{}
		_MainTex_ST("offset",Vector) = (1,1,0,0)
		//法线
		_BumpTex("Bump Texture",2D) = "bump"{}
        _BumpScale("Bump Scale",Float) = 1.0
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
				float4 tangent : TANGENT;//获取切线向量
            };

            struct outputData{
                float4 pos : SV_POSITION;
                float3 worldNormal : TEXCOORD0;
				float4 uv : TEXCOORD1; //zw保存_BumpTex的uv
				float3 worldPos : TEXCOORD2;
            	float3 lightDir : TEXCOORD3; //保存切线空间光线向量
			};

            fixed4 _BaseColor;
            float _Gloss;
			float _Intensity;
			sampler2D _MainTex;
			float4 _MainTex_ST;

            sampler2D _BumpTex;
            float _BumpScale;

            outputData vert (inputData v)
            {
                outputData o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                o.worldPos = mul(unity_ObjectToWorld,v.vertex).xyz;
				o.uv.xy = v.texcoord.xy *_MainTex_ST.xy + _MainTex_ST.zw;
				o.uv.zw = v.texcoord.xy;
				//bump的uv 放在zw
				//o.uv.zw = v.texcoord.xy *_BumpTex_ST.xy + _BumpTex_ST.zw;


				//副切线 = 法线X切线*方向参数w
				float3 binormal = cross(normalize(v.normal),normalize(v.tangent.xyz)) * v.tangent.w;

				//切线空间的变换矩阵 = (切线，副切线，法线)顺序排放
				float3x3 _2tangentSpace = float3x3(v.tangent.xyz,binormal,v.normal);

				//将光照向量从模型空间 转到 切线空间
				o.lightDir = mul(_2tangentSpace,ObjSpaceLightDir(v.vertex).xyz);

				return o;
            }

			fixed4 frag(outputData i):SV_TARGET{

				fixed3 tangentLightDir = normalize(i.lightDir);

				//从[0,1]映射到[-1,1],得到法线向量
				fixed3 tangentNormal = UnpackNormal(tex2D(_BumpTex,i.uv.zw));

				tangentNormal.xy = tangentNormal.xy*_BumpScale;

				//标准化
                fixed3 worldNormal = normalize(i.worldNormal);
                fixed3 worldLightDir = normalize(_WorldSpaceLightPos0.xyz);

				//reflect(光源，法线)
                fixed3 reflectDir = reflect(-worldLightDir,worldNormal);

				//相机
                fixed3 viewDir = normalize(UnityWorldSpaceViewDir(i.worldPos));

				float3 albedo = tex2D(_MainTex,i.uv).xyz * _BaseColor.xyz;

                fixed3 ambient = UNITY_LIGHTMODEL_AMBIENT.xyz * albedo;    //先获取环境光

				fixed bump = saturate(dot(tangentNormal,tangentLightDir));

				//saturate 相当于max加clam(0,1)
                fixed3 diffuse = _LightColor0.xyz * albedo * bump;

   				//利用Phong模型中的高光公式来计算高光
                fixed3 specular = _LightColor0.xyz * pow(saturate(dot(viewDir,reflectDir)),_Gloss);


                return fixed4(specular + ambient + diffuse,1.0);
            }

            ENDCG
        }
    }
}
