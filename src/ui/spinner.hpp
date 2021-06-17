
/*
 * Copyright (C) 2018 zfedoran
 *
 * Adapted from https://github.com/ocornut/imgui/issues/1901 
 *
 * Permission is hereby granted, free of charge, to any person obtaining 
 * a copy of this software and associated documentation files (the "Software"), 
 * to deal in the Software without restriction, including without limitation 
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 * and/or sell copies of the Software, and to permit persons to whom the 
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included 
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
 * THE SOFTWARE.
 */

#ifndef __US_GALLERY_SPINNER_HPP__
#define __US_GALLERY_SPINNER_HPP__

#include <numbers>

#include <imgui.h>
#include <imgui_internal.h>

namespace ImGui
{
  inline bool
  Spinner(const char* label,
	  float radius,
	  float thickness,
	  const ImU32& color)
  {
    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
      return false;
	
    ImGuiContext& g = *GImGui;
    auto const& style = g.Style;
    auto const id = window->GetID(label);
	
    ImVec2 pos = window->DC.CursorPos;
    ImVec2 size((radius )*2, (radius + style.FramePadding.y)*2);
	
    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    ItemSize(bb, style.FramePadding.y);
    if (!ItemAdd(bb, id))
      return false;
	
    // Render
    window->DrawList->PathClear();
    int num_segments = 30;
    int start = static_cast<int>(
      round(abs(ImSin(static_cast<float>(g.Time)*1.8f)*
		static_cast<float>(num_segments-5))));
	
    auto const pi     = std::numbers::pi_v<float>;
    const float a_min = pi*2.0f * (static_cast<float>(start))
      / static_cast<float>(num_segments);
    const float a_max = pi*2.0f * (static_cast<float>(num_segments)-3)
      / static_cast<float>(num_segments);

    const ImVec2 centre = ImVec2(pos.x+radius, pos.y+radius+style.FramePadding.y);
	
    for (int i = 0; i < num_segments; i++) {
      const float a = a_min + (static_cast<float>(i)
			       / static_cast<float>(num_segments)) * (a_max - a_min);
      window->DrawList->PathLineTo(
	ImVec2(centre.x + ImCos(a+static_cast<float>(g.Time*8)) * radius,
	       centre.y + ImSin(a+static_cast<float>(g.Time*8)) * radius));
    }
    window->DrawList->PathStroke(color, false, thickness);
    return true;
  }
}

#endif
