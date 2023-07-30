Shader "MyShader/Texture_GrabPass_TheWorld"
{
    Properties
    {
		_BaseColor("Base Color",Color) = (1.0,1.0,1.0,1.0)
		_MainTex("Main Texture",2D) = "white"{}
		 _Angle("_Angle",float) =  0
		 _R("R",float) =  0.5
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
			#define PI 3.1415926

            struct inputData{
                float4 vertex : POSITION;
                float3 normal : NORMAL;
				float4 texcoord : TEXCOORD0;
            };

            struct outputData{
                float4 pos : SV_POSITION;
				float4 uv : TEXCOORD0;
				float4 pos2 : TEXCOORD1;
            };

			sampler2D _GrabPassTexture;
			sampler2D _MainTex;
			fixed4 _BaseColor;
			float _Angle;
			float _R;

            outputData vert(inputData i)
            {
                outputData o;
                o.pos = UnityObjectToClipPos(i.vertex);
				o.pos2 = i.vertex;
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
			// RGB -> HSV 色彩空间
			float3 RGB2HSV(float3 c)
			{
				float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
				float4 p = lerp(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
				float4 q = lerp(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
				float d = q.x - min(q.w, q.y);
				float e = 1.0e-10;
				return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
			}

			//HSV -> RGB
			float3 HSV2RGB(float3 c)
			{
				float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
				float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
				return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
			}

			float3 CausticTriTwist(float2 uv,float time )
			{
				const int MAX_ITER = 5;
				float2 p = fmod(uv*PI,PI )-250.0;//1.空间划分

				float2 i = float2(p);
				float c = 1.0;
				float inten = .005;

				for (int n = 0; n < MAX_ITER; n++) //3.多层叠加
				{
					float t = time * (1.0 - (3.5 / float(n+1)));
					i = p + float2(cos(t - i.x) + sin(t + i.y), sin(t - i.y) + cos(t + i.x));//2.空间扭曲
					c += 1.0/length(float2(p.x / (sin(i.x+t)/inten),p.y / (cos(i.y+t)/inten)));//集合操作avg
				}

				c /= float(MAX_ITER);
				c = 1.17-pow(c, 1.4);//4.亮度调整
				float val = pow(abs(c), 8.0);
				return val;
			}

			fixed4 frag(outputData i):SV_TARGET{





				fixed3 albedo =tex2D(_MainTex,i.uv.xy);



				// if( distance(i.uv,float2(0.5,0.5)) > _R)
				if( distance(i.pos2.xy,float2(0.5,0.5)) > _R)
				{
					return float4(tex2D(_GrabPassTexture,i.uv.zw).xyz *_BaseColor *albedo,1);
				}

				return float4(tex2D(_GrabPassTexture,i.uv.xy).xyz *_BaseColor *albedo,1);

				i.uv.xy=i.uv.zw;
				//uv
				float2 uv = float2(i.uv.x-0.5,i.uv.y-0.5);
				float f = distance(uv,float2(0,0));
				float s = sin(lerp(0,_Angle,f));
				float c = cos(lerp(0,_Angle,f));
				// -c s
				// s c

				uv = float2(-(-uv.x*c+uv.y*s),uv.x*s+uv.y*c);

				uv = float2(uv.x+0.5,uv.y+0.5);
				// fixed4 col = tex2D(_MainTex, i.uv.zw);




				// float4 time = _Time;
                // float2 uvA = (i.uv0+(time.r*float2(_noise_A_U,_noise_A_V))*float2(1,1));//第一张图uv动画
                // float4 A = tex2D(_noise_A,TRANSFORM_TEX(uvA, _noise_A));
                // float2 uvB = (i.uv0+(time.r*float2(_noise_B_U,_noise_B_V))*float2(1,1));//第二张图uv动画
                // float4 B = tex2D(_noise_B,TRANSFORM_TEX(uvB, _noise_B));
                // float2 MainTexUV = ((i.uv0+((A.r+B.r-1)*_raodong_V))+(time.r*float2(_mianTex_U,_mianTex_V))*float2(1,1));//主贴图的扭曲uv + uv动画
                // float2 AlphaUV = i.uv0+((A.r+B.r-1)*_raodong_V);//另外加上一扭曲贴图的通道

                // float4 main_tex = tex2D(_main_tex,TRANSFORM_TEX(MainTexUV, _main_tex));//主贴图扭曲
                // float4 alpha_tex = tex2D(_tex_alpha,TRANSFORM_TEX(AlphaUV, _tex_alpha));//通道图的扭曲
                // float3 finalColor = (_main_V*((_Color.rgb*main_tex.rgb)*((alpha_tex.rgb*_alpha_color.rgb)*_alpha_v)));
                // fixed4 finalRGBA = fixed4(finalColor,1);
                // UNITY_APPLY_FOG(i.fogCoord, finalRGBA);


				fixed3 color = tex2D(_GrabPassTexture,uv.xy).xyz *_BaseColor *albedo;


				//颜色变化
				float3 hsvColor = RGB2HSV(color);
				hsvColor.x += lerp(0,0.2,sin( UNITY_TWO_PI * frac(_Time.y *0.5)));
				hsvColor.x = frac(hsvColor.x);

				hsvColor = HSV2RGB(hsvColor);

				//反色
				//hsvColor =  1 - hsvColor;



				// float2 uv =  o.uv;
				// float time = _Time.y;
				// float val = CausticTriTwist(uv,time);//替换相应的函数即可
				// return float3(val,val,val);

				return float4 (hsvColor,1);
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
