%{
#include "containers/2DArrayGPU.h"
%}

%inline %{
	struct AN2DRow {
		ANNGPGPU::F2DArray *g;
		int    y;

		// These functions are used by Python to access sequence types (lists, tuples, ...)
		float __getitem__(int x) {
			return g->GetValue(x, y);
		}

		void __setitem__(int x, float val) {
			g->SetValue(x, y, val);
		}
	};
%}

%include "containers/2DArrayGPU.h"  

%extend ANNGPGPU::F2DArray {
	AN2DRow __getitem__(int y) {
		AN2DRow r;
		r.g = self;
		r.y = y;
		return r;
	}
};
