
/*
 * Copyright (C) 2021  Ray Kim
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <algorithm>

#include <imgui.h>
#include <imgui-SFML.h>

#include "parameter_value.hpp"
#include "../custom_image_processing.hpp"

namespace usdg
{
  ParameterValue::
  ParameterValue()
    : _param_names(custom_ip_parameter_names()),
      _param_transformed(_param_names.size(), 0.0)
  { }

  void
  ParameterValue::
  render()
  {
    if(ImGui::Begin("ParameterValue"))
    {
      for (size_t i = 0; i < _param_names.size(); ++i)
      {
	ImGui::LabelText(_param_names[i].c_str(), "%.2f", _param_transformed[i]);
      }
    }
    ImGui::End();
  }

  void
  ParameterValue::
  update_parameter(blaze::DynamicVector<double> const& param)
  {
    _param_transformed =  custom_ip_transform_range(param);
  }
}
